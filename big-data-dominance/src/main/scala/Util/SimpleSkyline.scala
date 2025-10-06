package Util

import Util.Domination.isDominated

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.{break, breakable}

object SimpleSkyline {

  def calculate(partition: Iterator[Array[Double]]): Iterator[Array[Double]] = {
    var partitionList = partition.toList
    var i = 0
    var listLength = partitionList.length
    while (i < listLength - 1) {
      var k = i + 1
      while (k < listLength) {
        if (isDominated(partitionList(i), partitionList(k))) {
          partitionList = partitionList.take(k) ++ partitionList.drop(k + 1)
          k -= 1
          listLength -= 1
        } else if (isDominated(partitionList(k), partitionList(i))) {
          partitionList = partitionList.take(i) ++ partitionList.drop(i + 1)
          listLength -= 1
          i -= 1
          k = listLength
        }
        k += 1
      }
      i += 1
    }
    partitionList.toIterator
  }

  def calculateBlockNested(partition: Iterator[Array[Double]]): Iterator[Array[Double]] = {
    val candidates = ArrayBuffer[Array[Double]]()
    while (partition.hasNext) {
      val current = partition.next
      var addFlag = true
      val losers = ArrayBuffer[Array[Double]]()
      breakable {
        for (candidate <- candidates) {
          if (isDominated(candidate, current)) {
            addFlag = true
            break()
          } else if (isDominated(current, candidate))
            losers += candidate
        }
      }
      losers.foreach(candidates.-=)
      if (addFlag) candidates += current
    }
    candidates.toIterator
  }

}
