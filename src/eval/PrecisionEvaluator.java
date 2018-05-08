package eval;

import java.util.List;

/**
 * Created by zcy on 2018/4/27.
 */
public class PrecisionEvaluator extends AbstractEvaluator {
    @Override
    public double evaluateOneTest(List<Integer> rankList, int gtItem) {
        for (int i = 0; i < rankList.size(); i++) {
            int item = rankList.get(i);
            if (item == gtItem)
                return 1.0 / (i + 1);//todo ???
        }
        return 0;
    }
}
