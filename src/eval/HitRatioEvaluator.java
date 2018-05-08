package eval;

import java.util.List;

/**
 * Created by zcy on 2018/4/27.
 */
public class HitRatioEvaluator extends AbstractEvaluator {
    /**
     * Compute Hit Ratio.
     *
     * @param rankList A list of ranked item IDs
     * @param gtItem   The ground truth item.
     * @return Hit ratio.
     */
    @Override
    public double evaluateOneTest(List<Integer> rankList, int gtItem) {
        for (int item : rankList) {
            if (item == gtItem) return 1;
        }
        return 0;
    }
}
