package Util

import org.apache.spark.Partitioner

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.control.Breaks.{break, breakable}

class AngularPartitioner(numberOfPartitions: Int, dimension: Int) extends Partitioner {

  override def numPartitions: Int = numberOfPartitions

  if (numberOfPartitions < 1)
    throw new Exception("Number of partitions cannot be smaller than 1.")
  if (dimension < 2)
    throw new Exception("Cannot partition one-dimensional RDD.")
  if (numberOfPartitions > Math.pow(3, dimension - 1) || numberOfPartitions % 3 != 0)
    throw new Exception("Cannot split dimension into more than 3 parts.")

  val partitionRange: List[List[Double]] = makePartitionRange()

  def makePartitionRange(): List[List[Double]] = {
    var partitionRange = ListBuffer[List[Double]]()
    var n = numPartitions
    for (_ <- 0 until dimension - 1) {
      var range = List[Double]()
      if (n == 1)
        range = List(0, 90)
      else if (n == 2) {
        range = List(0, 45, 90)
        n /= 2
      } else if (n > 2) {
        range = List(0, 30, 60, 90)
        n /= 3
      }
      partitionRange += range
    }
    partitionRange.toList
  }

  def makeKey(point: Array[Double]): Int = {
    if (point.length != dimension)
      throw new Exception("Invalid key.")
    if (numPartitions == 1)
      return 0

    var angularCoord = ArrayBuffer[Double]()
    for (i <- 0 until point.length - 1) {
      // add one to avoid division by zero error
      val toDivideBy = point(i) + 1
      var squaredSum = 0.0
      for (j <- i + 1 until point.length)
        squaredSum += Math.pow(point(j), 2)
      val coord = Math.toDegrees(Math.atan(Math.sqrt(squaredSum) / toDivideBy))
      angularCoord += coord
    }

    var pointPartition = 0
    var len = 1
    var n = numPartitions
    breakable {
      for (i <- angularCoord.indices) {
        if (n == 1) {
          break
        }
        pointPartition += len *
          (partitionRange(i)
            .map(lim => lim > angularCoord(i))
            .zipWithIndex
            .filter(_._1)
            .head._2 - 1)
        val l = partitionRange(i).length - 1
        len *= l
        n /= l
      }
    }
    pointPartition
  }

  override def getPartition(key: Any): Int = key.asInstanceOf[Int]

}
