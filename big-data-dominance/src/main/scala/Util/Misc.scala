package Util

import scala.collection.mutable.ArrayBuffer

object Misc {

  def filterDominationPartition(partition: Iterator[(Array[Double], Int)], k: Int = 20): Iterator[Array[Double]] = {
    // Works only with sorted domination score partitions
    partition.toList.zipWithIndex.filter(_._2 < k).map(_._1._1).toIterator
  }

  def getMinValues(partition: Iterator[Array[Double]]): Iterator[Array[Double]] = {
    val partitionList = partition.toArray.transpose
    val minValues = ArrayBuffer[Double]()
    for (i <- partitionList.indices)
      minValues += partitionList(i).min
    Iterator(minValues.toArray)
  }

  def fillEmpty(partition: Iterator[Array[Double]], dim: Int): Iterator[Array[Double]] = {
    if (partition.isEmpty)
      return Iterator(Array.fill(dim)(Double.PositiveInfinity))
    partition
  }

  def applyMinValue(partition: Iterator[Array[Double]], minVal: Array[Double], subtract: Boolean = true):
  Iterator[Array[Double]] = {
    val newPartition = ArrayBuffer[Array[Double]]()
    while (partition.hasNext) {
      val x = partition.next
      for (i <- x.indices)
        if (subtract)
          x(i) -= minVal(i)
        else
          x(i) += minVal(i)
      newPartition += x
    }
    newPartition.toIterator
  }

}
