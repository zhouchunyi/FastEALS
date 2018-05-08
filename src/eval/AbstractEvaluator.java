package eval;

import java.util.List;

/**
 * evaluate for one test
 * 这里groundTruth只有一个
 * Created by zcy on 2018/4/27.
 */
public abstract class AbstractEvaluator {
    protected String evaluatorName;

    public AbstractEvaluator() {
        evaluatorName = getClass().getName();
    }

    /**
     * @param rankList top-k rankList
     * @param gtItem   GroundTruth item
     */
    public abstract double evaluateOneTest(List<Integer> rankList, int gtItem);

    public String getEvaluatorName() {
        return evaluatorName;
    }
}
