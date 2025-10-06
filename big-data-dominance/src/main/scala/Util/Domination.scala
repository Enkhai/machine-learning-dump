package Util

object Domination {

  def isDominated(left: Array[Double], right: Array[Double]): Boolean = {
    // left dominates right
    var same = true
    for (i <- left.indices) {
      if (left(i) > right(i))
        return false
      else if (same && left(i) < right(i))
        same = false
    }
    !same
  }

  def calculateDomination(partition: Iterator[Array[Double]]): Iterator[(Array[Double], Int)] = {
    val buffer = partition.toBuffer[Array[Double]].map(x => (x, 0))
    for (i <- 0 until buffer.length - 1) {
      for (j <- i + 1 until buffer.length) {
        if (isDominated(buffer(i)._1, buffer(j)._1))
          buffer(i) = (buffer(i)._1, buffer(i)._2 + 1)
        else if (isDominated(buffer(j)._1, buffer(i)._1))
          buffer(j) = (buffer(j)._1, buffer(j)._2 + 1)
      }
    }
    buffer.sortBy(-_._2).toIterator
  }

}
