package eval;

import java.util.List;

/**
 * 只有一条test的时候，precision和recall是否一样？？？
 * Created by zcy on 2018/4/27.
 */
public class RecallEvaluator extends AbstractEvaluator {
    @Override
    public double evaluateOneTest(List<Integer> rankList, int gtItem) {
        return 0;
    }
}
