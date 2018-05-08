package eval;

import java.util.List;

/**
 * Created by zcy on 2018/4/27.
 */
public class NDCGEvaluator extends AbstractEvaluator {
    /**
     * Compute NDCG of a list of ranked items.
     * See http://recsyswiki.com/wiki/Discounted_Cumulative_Gain
     * NDCG关注排序，加权排序
     * 由于这里只有一个GT，因此退化为2^r(l)/log(i+2)，其中r(l)=1
     *
     * @param rankList a list of ranked item IDs
     * @param gtItem   The ground truth item.
     * @return NDCG.
     */
    @Override
    public double evaluateOneTest(List<Integer> rankList, int gtItem) {
        for (int i = 0; i < rankList.size(); i++) {
            int item = rankList.get(i);
            if (item == gtItem)
                return Math.log(2) / Math.log(i + 2);
        }
        return 0;
    }
}
