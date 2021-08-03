/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.python

import java.io.File

import com.intel.oap.expression._
import com.intel.oap.vectorized._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.python.{ChainedPythonFunctions, PythonEvalType}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning}
import org.apache.spark.sql.execution.{GroupedIterator, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.util.Utils

/**
 * Physical node for aggregation with group aggregate Pandas UDF.
 *
 * This plan works by sending the necessary (projected) input grouped data as Arrow record batches
 * to the python worker, the python worker invokes the UDF and sends the results to the executor,
 * finally the executor evaluates any post-aggregation expressions and join the result with the
 * grouped key.
 */
case class ColumnarAggregateInPandasExec(
    groupingExpressions: Seq[NamedExpression],
    udfExpressions: Seq[PythonUDF],
    resultExpressions: Seq[NamedExpression],
    child: SparkPlan)
  extends UnaryExecNode {
    
  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "output_batches"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "input_batches"),
    "processTime" -> SQLMetrics.createTimingMetric(sparkContext, "totaltime_arrow_udf"))
  
  buildCheck()

  override val output: Seq[Attribute] = resultExpressions.map(_.toAttribute)

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def producedAttributes: AttributeSet = AttributeSet(output)

  override def supportsColumnar = true
    
  val sessionLocaTimeZone = conf.sessionLocalTimeZone
  val pythonRunnerConf = ArrowUtils.getPythonRunnerConfMap(conf)

  override def requiredChildDistribution: Seq[Distribution] = {
    if (groupingExpressions.isEmpty) {
      AllTuples :: Nil
    } else {
      ClusteredDistribution(groupingExpressions) :: Nil
    }
  }

  private def collectFunctions(udf: PythonUDF): (ChainedPythonFunctions, Seq[Expression]) = {
    udf.children match {
      case Seq(u: PythonUDF) =>
        val (chained, children) = collectFunctions(u)
        (ChainedPythonFunctions(chained.funcs ++ Seq(udf.func)), children)
      case children =>
        // There should not be any other UDFs, or the children can't be evaluated directly.
        assert(children.forall(_.find(_.isInstanceOf[PythonUDF]).isEmpty))
        (ChainedPythonFunctions(Seq(udf.func)), udf.children)
    }
  }

  def buildCheck(): Unit = {
    val (pyFuncs, inputs) = udfs.map(collectFunctions).unzip
    val allInputs = new ArrayBuffer[Expression]
    val dataTypes = new ArrayBuffer[DataType]
    val argOffsets = inputs.map { input =>
      input.map { e =>
        if (allInputs.exists(_.semanticEquals(e))) {
          allInputs.indexWhere(_.semanticEquals(e))
        } else {
          allInputs += e
          dataTypes += e.dataType
          allInputs.length - 1
        }
      }.toArray
    }.toArray
    ColumnarProjection.buildCheck(child.output, allInputs.toSeq)
  }

  override def requiredChildOrdering: Seq[Seq[SortOrder]] =
    Seq(groupingExpressions.map(SortOrder(_, Ascending)))

  override protected def doExecute(): RDD[InternalRow] = {
    throw new NotImplementedError("ColumnarAggregateInPandasExec only support doExecute for ColumnarBased format")
  }

  protected def evaluateColumnar(
    funcs: Seq[ChainedPythonFunctions],
    argOffsets: Array[Array[Int]],
    iter: Iterator[ColumnarBatch],
    schema: StructType,
    context: TaskContext): Iterator[ColumnarBatch] = {

    val outputTypes = output.drop(child.output.length).map(_.dataType)

    // Use Coalecse to improve performance in future
    val columnarBatchIter = new ColumnarArrowPythonRunner(
      funcs,
      PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
      argOffsets,
      schema,
      sessionLocalTimeZone,
      pythonRunnerConf).compute(iter, context.partitionId(), context)

    columnarBatchIter.map { batch =>
      val actualDataTypes = (0 until batch.numCols()).map(i => batch.column(i).dataType())
      assert(outputTypes == actualDataTypes, "Invalid schema from arrow_udf: " +
        s"expected ${outputTypes.mkString(", ")}, got ${actualDataTypes.mkString(", ")}")
      batch
    }
  }

  protected override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = longMetric("numOutputRows")
    val numOutputBatches = longMetric("numOutputBatches")
    val numInputBatches = longMetric("numInputBatches")
    val procTime = longMetric("processTime")

    val inputRDD = child.executeColumnar()


    val (pyFuncs, inputs) = udfExpressions.map(collectFunctions).unzip

    // Filter child output attributes down to only those that are UDF inputs.
    // Also eliminate duplicate UDF inputs.
    val allInputs = new ArrayBuffer[Expression]
    val dataTypes = new ArrayBuffer[DataType]
    val argOffsets = inputs.map { input =>
      input.map { e =>
        if (allInputs.exists(_.semanticEquals(e))) {
          allInputs.indexWhere(_.semanticEquals(e))
        } else {
          allInputs += e
          dataTypes += e.dataType
          allInputs.length - 1
        }
      }.toArray
    }.toArray

    // Schema of input rows to the python runner
    val projector = ColumnarProjection.create(child.output, allInputs.toSeq)
    val projected_ordinal_list = projector.getOrdinalList
    val schema = StructType(dataTypes.zipWithIndex.map { case (dt, i) =>
      StructField(s"_$i", dt)
    }.toSeq)
    val input_cb_cache = new ArrayBuffer[ColumnarBatch]()

    // Map grouped rows to ArrowPythonRunner results, Only execute if partition is not empty
    inputRDD.mapPartitionsInternal { iter => if (iter.isEmpty) iter else {
      val groupedColumnarBatchIter = if (groupingExpressions.isEmpty) {
        // Use an empty columnarBatch as a place holder for the grouping key
        Iterator((new ColumnarBatch(Array[ColumnVector]()), iter))
      } else {
        // GroupedIter will grouping by groupingExpressions
        ColumnarGroupedIterator(iter, groupingExpressions, child.output)
      }.map { case (key, inpur_cb_iter) => 
          (key, inpur_cb_iter.map { input_cb => 
            // 1. doing projection to input
            val valueVectors = (0 until input_cb.numCols).toList.map(i =>
                  input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].getValueVector())
            if (projector.needEvaluate) {
              val projectedInput = projector.evaluate(input_cb.numRows, valueVectors)
              new ColumnarBatch(projectedInput.toArray, input_cb.numRows)
            } else {
              // for no-need project evaluate, do another retain to align with projected new CB
              (0 until input_cb.numCols).foreach(i => {
                input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain()
              })
              new ColumnarBatch(projected_ordinal_list.toArray.map(i => input_cb.column(i)), input_cb.numRows)
            }
          }.map { batch =>
            val actualDataTypes = (0 until batch.numCols()).map(i => batch.column(i).dataType())
            assert(dataTypes == actualDataTypes, "Invalid schema for arrow_udf: " +
              s"expected ${dataTypes.mkString(", ")}, got ${actualDataTypes.mkString(", ")}")
            batch
          })
      }.map { case (groupingKey, grouped_cb_iter) =>
        input_cb_cache += groupingKey
        // Because grouped_cb is either created or retained, we need to close them here 
        new CloseableColumnBatchIterator(grouped_cb_iter)
      }

      val context = TaskContext.get()

      // The queue used to buffer input rows so we can drain it to
      // combine input with output from Python.
      context.addTaskCompletionListener[Unit] { _ =>
        projector.close
        input_cb_cache.foreach(_.close)
      }

      val outputColumnarBatchIterator = evaluateColumnar(
        pyFuncs, argOffsets, groupedColumnarBatchIter, schema, context)

      val joinedAttributes =
        groupingExpressions.map(_.toAttribute) ++ udfExpressions.map(_.resultAttribute)
      val resultProj = ColumnarProjection.create(joinedAttributes, resultExpressions)
      
      new CloseableColumnBatchIterator(
        outputColumnarBatchIterator.zipWithIndex.map { case (output_cb, batchId) =>
          val input_cb = input_cb_cache(batchId)
          // retain for input_cb since we are passing it to next operator
          (0 until input_cb.numCols).foreach(i => {
            input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain()
          })
          val joinedVectors = (0 until input_cb.numCols).toArray.map(i => input_cb.column(i)) ++ (0 until output_cb.numCols).toArray.map(i => output_cb.column(i))
          val numRows = input_cb.numRows
          numOutputBatches += 1
          numOutputRows += numRows
          procTime += (System.nanoTime() - start_time) / 1000000
          new ColumnarBatch(joinedVectors, numRows)
          // below is for in case there will be some scala projection in demand
          /*val valueVectors = joinedVectors.toList.map(_.asInstanceOf[ArrowWritableColumnVector].getValueVector())
          val projectedOutput = resultProj.evaluate(numRows, valueVectors)
          new ColumnarBatch(projectedOutput.toArray.map(_.asInstanceOf[ColumnVector]), numRows)*/
        }
      )
    }}
  }
}
