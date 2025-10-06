import Util.AngularPartitioner
import Util.Domination.calculateDomination
import Util.Misc.{applyMinValue, fillEmpty, filterDominationPartition, getMinValues}
import org.apache.log4j._
import org.apache.spark.{SparkConf, SparkContext}

object HighestDominance {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark.SparkContext").setLevel(Level.WARN)

    val sparkConf = new SparkConf()
      .setMaster("local[4]")
      .setAppName("HighestDominance")

    val sc = new SparkContext(sparkConf)

    val currentDir = System.getProperty("user.dir")
    val inputFile = "file://" + currentDir + "/datasets/gaussian_size1000_dim2.csv"
    val outputDir = "file://" + currentDir + "/output"

    var points = sc.textFile(inputFile)
      .map(x => x.split(","))
      .map(x => x.map(y => y.toDouble))

    val minVal = points.mapPartitions(getMinValues)
      .coalesce(1)
      .mapPartitions(getMinValues)
      .collect
      .head
    val numPartitions = 3
    val partitioner = new AngularPartitioner(numPartitions, minVal.length)

    points = points
      .mapPartitions(partition => applyMinValue(partition, minVal))
      .mapPartitions(_.map(x => (partitioner.makeKey(x), x)))
      .partitionBy(partitioner)
      .mapPartitions(_.map(_._2))
      .mapPartitions(partition => fillEmpty(partition, minVal.length))

    val result = points.mapPartitions(calculateDomination)
      .mapPartitions(partition => filterDominationPartition(partition))
      .coalesce(1)
      .mapPartitions(calculateDomination)
      .mapPartitions(partition => filterDominationPartition(partition))
      .mapPartitions(partition => applyMinValue(partition, minVal, subtract = false))

    result.map(_.mkString(", ")).saveAsTextFile(outputDir)

    sc.stop()
  }
}