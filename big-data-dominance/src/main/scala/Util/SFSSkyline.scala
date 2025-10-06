package Util

import Util.Domination.isDominated

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.{break, breakable}

object SFSSkyline {

  // Simple implementation

  def calculate(partitionArray: Array[Array[Double]]): Iterator[Array[Double]] = {
    val buffer = ArrayBuffer[Array[Double]]()
    buffer += partitionArray(0)
    for (i <- 1 until partitionArray.length) {
      var j = 0
      var toBeAdded = true
      breakable {
        while (j < buffer.length) {
          if (isDominated(partitionArray(i), buffer(j))) {
            buffer.remove(j)
            j -= 1
          } else if (isDominated(buffer(j), partitionArray(i))) {
            toBeAdded = false
            break()
          }
          j += 1
        }
      }
      if (toBeAdded)
        buffer += partitionArray(i)
    }
    buffer.toIterator
  }

  def addScoreAndCalculate(partition: Iterator[Array[Double]]): Iterator[Array[Double]] = {
    calculate(
      partition.map(x => {
        var sum = 0.0
        for (i <- x.indices)
          sum += math.log(x(i) + 1)
        (x, sum)
      }).toArray
        .sortBy(-_._2)
        .map(_._1)
    )
  }

  // Domination score tuple implementation

  def calculate(partitionArray: Array[(Array[Double], Int)]): Iterator[(Array[Double], Int)] = {
    val buffer = ArrayBuffer[(Array[Double], Int)]()
    buffer += partitionArray(0)
    for (i <- 1 until partitionArray.length) {
      var j = 0
      var toBeAdded = true
      breakable {
        while (j < buffer.length) {
          if (isDominated(partitionArray(i)._1, buffer(j)._1)) {
            buffer.remove(j)
            j -= 1
          } else if (isDominated(buffer(j)._1, partitionArray(i)._1)) {
            toBeAdded = false
            break()
          }
          j += 1
        }
      }
      if (toBeAdded)
        buffer += partitionArray(i)
    }
    buffer.toIterator
  }

  def addScoreAndCalculateWithDomination(partition: Iterator[(Array[Double], Int)]): Iterator[(Array[Double], Int)] = {
    calculate(
      partition.map(x => {
        var sum = 0.0
        for (i <- x._1.indices)
          sum += math.log(x._1(i) + 1)
        (x, sum)
      }).toArray
        .sortBy(-_._1._2)
        .map(_._1)
    )
  }
}
