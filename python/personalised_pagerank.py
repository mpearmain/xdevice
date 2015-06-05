/**
  * Calculate a personalized PageRank around the given user, and return
* a list of the nodes with the highest personalized PageRank scores.
*
  * @return A list of (node, probability of landing at this node after
                       *         running a personalized PageRank for K iterations) pairs.
*/
  def pageRank(user: Int): List[(Int, Double)] = {
    // This map holds the probability of landing at each node, up to the
    // current iteration.
    val probs = Map[Int, Double]()
    probs(user) = 1 // We start at this user.

    val pageRankProbs = pageRankHelper(start, probs, NumPagerankIterations)
    pageRankProbs.toList
    .sortBy { -_._2 }
    .filter { case (node, score) =>
      !getFollowings(user).contains(node) && node != user
    }
    .take(MaxNodesToKeep)
  }

  /**
    * Simulates running a personalized PageRank for one iteration.
  *
    * Parameters:
    * start - the start node to calculate the personalized PageRank around
  * probs - a map from nodes to the probability of being at that node at
  *         the start of the current iteration
  * numIterations - the number of iterations remaining
  * alpha - with probability alpha, we follow a neighbor; with probability
  *         1 - alpha, we teleport back to the start node
  *
    * @return A map of node -> probability of landing at that node after the
  *         specified number of iterations.
  */
    def pageRankHelper(start: Int, probs: Map[Int, Double], numIterations: Int,
                       alpha: Double = 0.5): Map[Int, Double] = {
                         if (numIterations <= 0) {
                           probs
                         } else {
                           // Holds the updated set of probabilities, after this iteration.
                           val probsPropagated = Map[Int, Double]()

                           // With probability 1 - alpha, we teleport back to the start node.
                           probsPropagated(start) = 1 - alpha

                           // Propagate the previous probabilities...
                           probs.foreach { case (node, prob) =>
                             val forwards = getFollowings(node)
                             val backwards = getFollowers(node)

                             // With probability alpha, we move to a follower...
                             // And each node distributes its current probability equally to
                             // its neighbors.
                             val probToPropagate = alpha * prob / (forwards.size + backwards.size)
                             (forwards.toList ++ backwards.toList).foreach { neighbor =>
                               if (!probsPropagated.contains(neighbor)) {
                                 probsPropagated(neighbor) = 0
                               }
                             probsPropagated(neighbor) += probToPropagate
                             }
                           }

                           pageRankHelper(start, probsPropagated, numIterations - 1, alpha)
                         }
                       }