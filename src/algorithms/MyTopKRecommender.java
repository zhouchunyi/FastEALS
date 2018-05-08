//package algorithms;
//
//import data_structure.DenseMatrix;
//import data_structure.DenseVector;
//import data_structure.Rating;
//import data_structure.SparseMatrix;
//import utils.CommonUtils;
//import utils.Printer;
//
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//
///**
// * Created by zcy on 2018/4/27.
// */
//public abstract class MyTopKRecommender {
//    /**
//     * Name of this current method
//     */
//    public String method_name;
//
//    /**
//     * The number of users.
//     */
//    public int userCount;
//    /**
//     * The number of items.
//     */
//    public int itemCount;
//    /**
//     * Rating matrix of training set. Users by Items.
//     */
//    public SparseMatrix trainMatrix;
//    /**
//     * Test ratings. For showing progress only.
//     */
//    public ArrayList<Rating> testRecords;
//
//    /**
//     * Position to cutoff.
//     */
//    public int topK = 100;
//    /**
//     * Number of threads to run the model (if multi-thread implementation).
//     */
//    public int threadNum = 1;
//
//    /**
//     * Evaluation for each user (offline eval) or test instance (online eval).
//     */
////    public DenseVector hits;
////    public DenseVector ndcgs;
////    public DenseVector precs;
//    public Map<String, DenseVector> evaluatorMap;
//    public int maxIterOnline = 1;
//
//    public boolean ignoreTrain = false; // ignore train items when generating topK list
//
//    public MyTopKRecommender() {
//        //子类会变更至自己的类名
//        this.method_name = getClass().getName();
//        this.evaluatorMap = new HashMap<>();
//    }
//
//
//    public MyTopKRecommender(SparseMatrix trainMatrix,
//                             ArrayList<Rating> testRecords, int topK, int threadNum) {
//        this();
//        this.trainMatrix = new SparseMatrix(trainMatrix);
//        this.testRecords = new ArrayList<Rating>(testRecords);
//        this.topK = topK;
//        this.threadNum = threadNum;
//
//        this.userCount = trainMatrix.length()[0];
//        this.itemCount = trainMatrix.length()[1];
//    }
//
//    /**
//     * Get the prediction score of user u on item i. To be overridden.
//     */
//    public abstract double predict(int u, int i);
//
//    /**
//     * Build the model.
//     */
//    public abstract void buildModel();
//
//    /**
//     * Update the model with a new observation.
//     */
//    public abstract void updateModel(int u, int i);
//
//    /**
//     * Show progress (evaluation) with current model parameters.
//     *
//     * @iter Current iteration
//     * @start Starting time of the iteration
//     * @testMatrix For evaluation purpose
//     */
//    public void showProgress(int iter, long start, ArrayList<Rating> testRecords) {
//        long end_iter = System.currentTimeMillis();
//        if (userCount == testRecords.size())  // leave-1-out eval
//            evaluate(testRecords);
//        else    // global split
//            evaluateOnline(testRecords, 100);
//        long end_eval = System.currentTimeMillis();
//
//        System.out.printf("Iter=%d[%s]: loss\t%.4f", iter, Printer.printTime(end_iter - start), loss());
//        this.evaluatorMap.entrySet().stream().forEach(evaluatorEntry -> {
//            System.out.printf(",\t" + evaluatorEntry.getKey() + "\t%.4f", evaluatorEntry.getValue().mean());
//        });
//        System.out.println(",\t" + Printer.printTime(end_eval - end_iter));
////        System.out.printf("Iter=%d[%s] <loss, hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f\t %.4f\t [%s]\n",
////                iter, Printer.printTime(end_iter - start), loss(),
////                hits.mean(), ndcgs.mean(), precs.mean(), Printer.printTime(end_eval - end_iter));
//    }
//
//    /**
//     * 设置当前recommender需要哪些evaluators
//     *
//     * @param evaluator_names 以逗号分隔
//     */
//    public void setEvaluators(String evaluator_names) {
//        String[] eval_names = evaluator_names.split(",");
//
//    }
//
//    /**
//     * Online evaluation (global split) by simulating the testing stream.
//     * todo 在这里加一些测试
//     *
//     * @param testRecords Test ratings that are sorted by time (old -> recent).
//     * @param interval    Print evaluation result per X iteration.
//     */
//    public void evaluateOnline(ArrayList<Rating> testRecords, int interval) {
//        int testCount = testRecords.size();
//
////        hits = new DenseVector(testCount);
////        ndcgs = new DenseVector(testCount);
////        precs = new DenseVector(testCount);
//
//        // break down the results by number of user ratings of the test pair
//        int intervals = 10;
//        int[] counts = new int[intervals + 1];
//        double[] hits_r = new double[intervals + 1];
//        double[] ndcgs_r = new double[intervals + 1];
//        double[] precs_r = new double[intervals + 1];
//
//        Long updateTime = (long) 0;
//        for (int i = 0; i < testCount; i++) {
//            // Check performance per interval:
//            if (i > 0 && interval > 0 && i % interval == 0) {
//                System.out.printf("%d: <hr, ndcg, prec> =\t %.4f\t %.4f\t %.4f\n",
//                        i, hits.sum() / i, ndcgs.sum() / i, precs.sum() / i);
//            }
//            // Evaluate model of the current test rating:
//            Rating rating = testRecords.get(i);
//            double[] res = this.evaluate_for_user(rating.userId, rating.itemId);
//            hits.set(i, res[0]);
//            ndcgs.set(i, res[1]);
//            precs.set(i, res[2]);
//
//            // statisitcs for break down
//            int r = trainMatrix.getRowRef(rating.userId).itemCount();
//            r = r > intervals ? intervals : r;
//            counts[r] += 1;
//            hits_r[r] += res[0];
//            ndcgs_r[r] += res[1];
//            precs_r[r] += res[2];
//
//            // Update the model
//            Long start = System.currentTimeMillis();
//            updateModel(rating.userId, rating.itemId);
//            updateTime += (System.currentTimeMillis() - start);
//        }
//
//        System.out.println("Break down the results by number of user ratings for the test pair.");
//        System.out.printf("#Rating\t Percentage\t HR\t NDCG\t MAP\n");
//        for (int i = 0; i <= intervals; i++) {
//            System.out.printf("%d\t %.2f%%\t %.4f\t %.4f\t %.4f \n",
//                    i, (double) counts[i] / testCount * 100,
//                    hits_r[i] / counts[i], ndcgs_r[i] / counts[i], precs_r[i] / counts[i]);
//        }
//
//        System.out.printf("Avg model update time per instance: %.2f ms\n", (float) updateTime / testCount);
//    }
//
//    protected ArrayList<Integer> threadSplit(int total, int threadNum, int t) {
//        ArrayList<Integer> res = new ArrayList<Integer>();
//        int start = (total / threadNum) * t;
//        int end = (t == threadNum - 1) ? total :
//                (total / threadNum) * (t + 1);
//        for (int i = start; i < end; i++)
//            res.add(i);
//        return res;
//    }
//
//    /**
//     * Offline evaluation (leave-1-out) for each user.
//     */
//    public void evaluate(ArrayList<Rating> testRecords) {
//        assert userCount == testRecords.size();
//        for (int u = 0; u < userCount; u++)
//            assert u == testRecords.get(u).userId;
//
//        hits = new DenseVector(userCount);
//        ndcgs = new DenseVector(userCount);
//        precs = new DenseVector(userCount);
//
//        // Run the evaluation multi-threads splitted by users
//        MyEvaluationThread[] threads = new MyEvaluationThread[threadNum];
//        for (int t = 0; t < threadNum; t++) {
//            ArrayList<Integer> users = threadSplit(userCount, threadNum, t);
//            threads[t] = new MyEvaluationThread(this, testRecords, users);
//            threads[t].start();
//        }
//
//        // Wait until all threads are finished.
//        for (int t = 0; t < threads.length; t++) {
//            try {
//                threads[t].join();
//            } catch (InterruptedException e) {
//                System.err.println("InterruptException was caught: " + e.getMessage());
//            }
//        }
//    }
//
//    /**
//     * Evaluation for a specific user with given GroundTruth item.
//     *
//     * @return: result[0]: hit ratio
//     * result[1]: ndcg
//     * result[2]: precision
//     */
//    protected double[] evaluate_for_user(int u, int gtItem) {
//        double[] result = new double[3];
//        HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
//        // Get the score of the test item first.
//        double maxScore = predict(u, gtItem);
//
//        // Early stopping if there are topK items larger than maxScore.
//        int countLarger = 0;
//        for (int i = 0; i < itemCount; i++) {
//            double score = predict(u, i);
//            map_item_score.put(i, score);
//
//            if (score > maxScore) countLarger++;
//            if (countLarger > topK) return result;    // early stopping
//        }
//
//        // Selecting topK items (does not exclude train items).
//        ArrayList<Integer> rankList = ignoreTrain ?
//                CommonUtils.TopKeysByValue(map_item_score, topK, trainMatrix.getRowRef(u).indexList()) :
//                CommonUtils.TopKeysByValue(map_item_score, topK, null);
//        result[0] = getHitRatio(rankList, gtItem);
//        result[1] = getNDCG(rankList, gtItem);
//        result[2] = getPrecision(rankList, gtItem);
//
//        return result;
//    }
//
//    /**
//     * Compute Hit Ratio.
//     *
//     * @param rankList A list of ranked item IDs
//     * @param gtItem   The ground truth item.
//     * @return Hit ratio.
//     */
//    public double getHitRatio(List<Integer> rankList, int gtItem) {
//        for (int item : rankList) {
//            if (item == gtItem) return 1;
//        }
//        return 0;
//    }
//
//    /**
//     * Compute NDCG of a list of ranked items.
//     * See http://recsyswiki.com/wiki/Discounted_Cumulative_Gain
//     *
//     * @param rankList a list of ranked item IDs
//     * @param gtItem   The ground truth item.
//     * @return NDCG.
//     */
//    public double getNDCG(List<Integer> rankList, int gtItem) {
//        for (int i = 0; i < rankList.size(); i++) {
//            int item = rankList.get(i);
//            if (item == gtItem)
//                return Math.log(2) / Math.log(i + 2);
//        }
//        return 0;
//    }
//
//    public double getPrecision(List<Integer> rankList, int gtItem) {
//        for (int i = 0; i < rankList.size(); i++) {
//            int item = rankList.get(i);
//            if (item == gtItem)
//                return 1.0 / (i + 1);
//        }
//        return 0;
//    }
//
//    // remove
//    public void runOneIteration() {
//    }
//
//    // remove
//    public double loss() {
//        return 0;
//    }
//
//    // remove
//    public void setUV(DenseMatrix U, DenseMatrix V) {
//    }
//
//
//}
//
//class MyEvaluationThread extends Thread {
//    MyTopKRecommender model;
//    ArrayList<Rating> testRecords;
//    ArrayList<Integer> users;
//
//    public MyEvaluationThread(MyTopKRecommender model, ArrayList<Rating> testRecords,
//                              ArrayList<Integer> users) {
//        this.model = model;
//        this.testRecords = testRecords;
//        this.users = users;
//    }
//
//    public void run() {
//        for (int u : users) {
//            double[] res = model.evaluate_for_user(u, testRecords.get(u).itemId);
//            model.hits.set(u, res[0]);
//            model.ndcgs.set(u, res[1]);
//            model.precs.set(u, res[2]);
//        }
//    }
//}
