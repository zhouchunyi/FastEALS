package algorithms;

import data_structure.DenseVector;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.CommonUtils;
import utils.Printer;

import java.util.*;

/**
 * Bandit-like recommender
 * evaluator: only top k -hits
 * Created by zcy on 2018/5/10.
 */
public abstract class OnlineTopOneRecommender {
    /*************************Data Structures******************************/
    /**
     * Rating matrix of training set. Users by Items.
     */
    public SparseMatrix trainMatrix;
    /**
     * Test ratings. For showing progress only.
     */
    public ArrayList<Rating> testRatings;
    /**
     * online Evaluation
     */
    public DenseVector hits;
    /**************************Properties*********************************/
    /**
     * Name of this current method
     */
    public String method_name;

    /**
     * The number of users.
     */
    public int userCount;
    /**
     * The number of items.
     */
    public int itemCount;
    /**
     * Position to cutoff.
     * todo default: recommend top-1 for user
     */
    public int topK = 1;
    /**
     * iterNum for online update, 实验表明1轮比较好
     */
    public int maxIterOnline = 1;
    /**
     * ignore train items when generating topK list
     * 忽略的是某user已经观察过的item，即不重复推荐
     */
    public boolean ignoreTrain = false;

    /**
     * for online evaluation (break down)
     * todo 新增，应该和item数量相关
     */
    public int intervalsTimes = 100;

    /**
     * 随机挑选item的随机数
     */
    public Random random;
    public int randomSeed = 1;

    public OnlineTopOneRecommender() {
        //子类会变更至自己的类名
        method_name = getClass().getName();
        random = new Random(randomSeed);
    }

    public OnlineTopOneRecommender(SparseMatrix trainMatrix,
                                   ArrayList<Rating> testRatings, int topK) {
        this();
        this.trainMatrix = new SparseMatrix(trainMatrix);
        this.testRatings = new ArrayList<Rating>(testRatings);
        this.topK = topK;

        this.userCount = trainMatrix.length()[0];
        this.itemCount = trainMatrix.length()[1];
    }

    /**
     * Get the prediction score of user u on item i. To be overridden.
     */
    public abstract double predict(int u, int i);

    /**
     * Bandit-like udpate
     * Update the model with a new observation.
     *
     * @param u      current userIndex
     * @param i      itemIndex that system recommends last round
     * @param reward the feedback user given in this round
     */
    public abstract void updateModel(int u, int i, double reward);

    /**
     * Offline Train
     */
    public abstract void buildModel();

    /**
     * todo recommend top-1 item for user
     */
//    public abstract void topOneItem();

    /**
     * Show progress (evaluation) with current model parameters.
     * only
     *
     * @param iter        current iteration
     * @param start       time of the iteration
     * @param testRatings
     */
    public void showProgress(int iter, long start, ArrayList<Rating> testRatings) {
        long end_iter = System.currentTimeMillis();

        evaluateOnline(testRatings, 100);
        long end_eval = System.currentTimeMillis();

        System.out.printf(new Date() + "\tIter=%d[%s] <loss, hit>:\t %.4f\t %.4f\t [%s]\n",
                iter, Printer.printTime(end_iter - start), loss(),
                hits.mean(), Printer.printTime(end_eval - end_iter));
    }

    public void evaluateOnline(ArrayList<Rating> testRatings, int interval) {
        int testCount = testRatings.size();
        //记录每一条test的评价结果
        hits = new DenseVector(testCount);

        // break down the results by number of user ratings of the test pair
        /**
         *在每轮测试后，根据该user在trainMatrix的非零项数量break down
         */
        int intervals = 10;
        int[] counts = new int[intervals + 1];
        double[] hits_r = new double[intervals + 1];

        ArrayList<Integer> recommendList;
        Long updateTime = (long) 0;
        int selectedItem;
        double hitRes;
        for (int i = 0; i < testCount; i++) {//逐条测试
            // Check performance per interval:
            if (i > 0 && interval > 0 && i % interval == 0) {
                //每个interval轮输出一次评价指标
                System.out.printf(new Date() + "\t%d: <hits> =\t %.4f\n",
                        i, hits.sum() / i);
            }
            // Evaluate model of the current test rating:
            Rating rating = testRatings.get(i);
            recommendList = recommendHighestScore(rating.userId);
            selectedItem = recommendList.get(random.nextInt(recommendList.size()));

            /**
             * 这里只用binary reward，即hit
             * 对比groundTruth确定hit
             */
            if (selectedItem == rating.itemId)
                hitRes = 1;
            else
                hitRes = 0;

//            double hitRes = this.evaluate_for_user(rating.userId, rating.itemId);
            hits.set(i, hitRes);

            // statisitcs for break down
            int r = trainMatrix.getRowRef(rating.userId).itemCount();//userId在train里面不为0的数量

            r = r > intervals * intervalsTimes ? intervals : r / intervalsTimes;//todo 修改过
            counts[r] += 1;
            hits_r[r] += hitRes;

            // Update the model
            Long start = System.currentTimeMillis();
            //todo updateModel 这里缺少top-1推荐
            updateModel(rating.userId, selectedItem, hitRes);
            updateTime += (System.currentTimeMillis() - start);
        }
        System.out.printf(new Date() + "\t%d: <hits> =\t %.4f\n", testCount, hits.sum() / testCount);
        System.out.println("Break down the results by number of user ratings for the test pair.");
        System.out.printf("#Rating\t Percentage\t HR\n");

        for (int i = 0; i <= intervals; i++) {
            System.out.printf("%d\t %.2f%%\t %.4f\n", i * intervalsTimes, (double) counts[i] / testCount * 100, hits_r[i] / counts[i]);
        }

        System.out.printf("Avg model update time per instance: %.2f ms\n", (float) updateTime / testCount);
    }

    /**
     * for online evaluation
     * 为所有item计算predict，推荐score最高的item
     * 可能存在相同score => 返回list
     * 由于需要遍历所有item => 耗时长
     *
     * @param userIndex
     * @return
     */
    protected ArrayList<Integer> recommendHighestScore(int userIndex) {
        ArrayList<Integer> hightestScoreList = new ArrayList<>();
        double maxScore = predict(userIndex, 0), score;
        hightestScoreList.add(0);
        for (int i = 1; i < itemCount; i++) {
            score = predict(userIndex, i);
            if (score > maxScore) {
                maxScore = score;
                hightestScoreList.clear();
                hightestScoreList.add(i);
            }
        }
        return hightestScoreList;
    }

//    protected double evaluate_for_user(int u, int gtItem) {
//        double hitRes = 0;
//        HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
//        // Get the score of the test item first.
//        double maxScore = predict(u, gtItem);//user对groundTruth的预测
//
//        // Early stopping if there are topK items larger than maxScore.
//        int countLarger = 0;//其他item预测的score超过groundTruth的个数
//        for (int i = 0; i < itemCount; i++) {
//            double score = predict(u, i);
//            map_item_score.put(i, score);
//
//            if (score > maxScore) countLarger++;
//            if (countLarger > topK) return hitRes;    // early stopping：groundTruth没有出现在topK列表中
//        }
//
//        // Selecting topK items (does not exclude train items).
//        ArrayList<Integer> rankList = ignoreTrain ?
//                CommonUtils.TopKeysByValue(map_item_score, topK, trainMatrix.getRowRef(u).indexList()) :
//                CommonUtils.TopKeysByValue(map_item_score, topK, null);
//        hitRes = getHitRatio(rankList, gtItem);
//
//        return hitRes;
//    }


    public double loss() {
        return 0;
    }
}
