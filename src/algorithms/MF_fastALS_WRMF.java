package algorithms;

import data_structure.DenseMatrix;
import data_structure.DenseVector;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.Printer;

import java.util.ArrayList;
import java.util.Date;

/**
 * Fast ALS for weighted matrix factorization (with imputation)
 *
 * @author Chunyi Zhou修改项：
 *         （1）将reg规格化参数具体分为了regUser和regItem；
 *         （2）***添加了参数设置部分；
 *         （3）confidence计算时的alpha参数可设置。
 */
public class MF_fastALS_WRMF extends TopKRecommender {
    /**
     * Model priors to set.
     */
    public int factors = 10;    // number of latent factors.
    public int maxIter = 500;    // maximum iterations.
    //    double reg = 0.01;    // regularization parameters
    public double regUser = 0.01, regItem = 0.01;//todo 通常regUser和regItem设为一样的
    public double c0 = 1;
    public double init_mean = 0;  // Gaussian mean for init V
    public double init_stdev = 0.01; // Gaussian std-dev for init V
    /**
     * parameter for confidence computation, paper中提到经验来说0.5效果不错，当前proj中xiangnanhe设置为0.4
     * todo Need Tuning
     */
    public float alpha = 0.4f;
    public float weightCoefficient = 4.0f;//confidence weight coefficient for WRMF(for weight_{ui} computing)

    /**
     * *************new*****************
     * 0：eALS MF; 1：WRMF; 2: both; 3：WRMF+观察到部分*popularity
     * [differences]
     * 0: only consider zero part(i.e. missing data)
     * 1: only consider non-zero part
     * 2: both consider zero and non-zero [this paper]
     * 3：WRMF + non-zero*popularity # 我们的尝试
     */

    private int WRMFJudge = 1;

    /**
     * Model parameters to learn
     */
    public DenseMatrix U;    // latent vectors for users : userNum*factors
    public DenseMatrix V;    // latent vectors for items : itemNum*factors

    /**
     * Caches
     */
    //SU和SV应该就是librec里面userFactorsCache和itemFactorsCache => numFactors*numFactors
    private DenseMatrix SU;//square U
    private DenseMatrix SV;//square V
    private double[] prediction_users, prediction_items;
    private double[] rating_users, rating_items;
    private double[] w_users, w_items;


    public boolean showProgress;
    public boolean showLoss;

    /**
     * todo 为了观察新旧用户的推荐效果
     * 仅出现在test中的user为newUser
     */
//    private boolean[] isNewUser;
//    private boolean isTrain;
    /**
     * weight是否在在线update过程中随着interaction变化
     */
    public boolean isOnlineWeightUpdate = false;

    /**
     * 权重参数
     */
    // 1.weight for each positive instance in trainMatrix => weights of all user-item pair (u,i)
    private SparseMatrix weights;
    // 2.weight for negative instances on item i.==> confidence that item i missed by users
    private double[] confidences;//原代码参数名为Wi
    /**
     * for WRMFJudge=3
     * 计算时不考虑confidence，但仍需要为item计算popularity，这里利用confidence_cache
     */
    private double[] confidences_cache;
    // 3.weight of new instance in online learning
    public double w_init = 1;


    public MF_fastALS_WRMF(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, int WRMFJudge,
                           int topK, int threadNum, int factors, int maxIter, double c0, float alpha, double regUser, double regItem,
                           double init_mean, double init_stdev, boolean showProgress, boolean showLoss) {
        super(trainMatrix, testRatings, topK, threadNum);
        this.factors = factors;
        this.maxIter = maxIter;
        this.c0 = c0;
        this.regUser = regUser;
        this.regItem = regItem;
        this.init_mean = init_mean;
        this.init_stdev = init_stdev;
        this.showLoss = showLoss;
        this.showProgress = showProgress;
        this.alpha = alpha;
        this.WRMFJudge = WRMFJudge;

        /**
         * new weights and confidences setting
         */
        long start = System.currentTimeMillis();
        System.out.println(new Date() + "\t" + method_name + " weights and confidences init start.");
        initConfidencesAndWeights();
        System.out.println(new Date() + "\t" + method_name + " weights and confidences init end." + Printer.printTime(start - System.currentTimeMillis()));


        // Init caches
        prediction_users = new double[userCount];
        prediction_items = new double[itemCount];
        rating_users = new double[userCount];
        rating_items = new double[itemCount];
        w_users = new double[userCount];
        w_items = new double[itemCount];

        // Init model parameters
        U = new DenseMatrix(userCount, factors);
        V = new DenseMatrix(itemCount, factors);
        //gaussian init
        U.init(init_mean, init_stdev);
        V.init(init_mean, init_stdev);
        initS();

        //todo init newUser array => false by default
//        isNewUser = new boolean[userCount];
//        isTrain = true;//由于train和test部分方法有重叠
    }

    /**
     * todo 根据WRMFJudge不同，对weights和confidencs给予不同赋值
     */
    private void initConfidencesAndWeights() {
        /**
         * Confidences Computation
         */
        confidences = new double[itemCount];//confidence用于计算向量，而confidence_cache用于weight计算
        confidences_cache = new double[itemCount];
        if (WRMFJudge == 0 || WRMFJudge == 2 || WRMFJudge == 3) {
            /**
             * c_i = c_0*{f^alpha_i/SUM[all items]{f^alpha_j}}
             * alphaPopularity_i = f^alpha_i = |R_i|
             *                   =>intuition: a miss on a popular item is more probable to be truly irrelevant to the user
             */
            double sum = 0, Z = 0;//sum是train中non-zero个数
            //p[itemIndex]非空的元素个数，当前即为train中non-zero的个数
            double[] p = new double[itemCount];
            for (int i = 0; i < itemCount; i++) {
                p[i] = trainMatrix.getColRef(i).itemCount();//trainMatrix.getColRef(i)获得第i列的Vector
//                System.out.println("item[" + i + "]=" + p[i]);
//                if(p[i]==0)
//                    System.out.println("item["+i+"] is new item");
                sum += p[i];
            }
            // convert p[i] to probability
            for (int i = 0; i < itemCount; i++) {
                p[i] /= sum;
                p[i] = Math.pow(p[i], alpha);
                Z += p[i];
            }
//             assign weight
            if (WRMFJudge == 0 || WRMFJudge == 2) {
                for (int i = 0; i < itemCount; i++)
                    confidences[i] = c0 * p[i] / Z;
            } else {//WRMFJudge==3
                for (int i = 0; i < itemCount; i++)
                    confidences_cache[i] = c0 * p[i] / Z;
            }

        }

        if (WRMFJudge == 1 || WRMFJudge == 3) {//对missing部分不做处理，confidence都设置为1
            for (int i = 0; i < itemCount; i++) {
                confidences[i] = 1;
            }
        }

        /**
         * Weights Computation
         * 仅对有数据的部分
         */
        weights = new SparseMatrix(userCount, itemCount);
        // By default, the weight for positive instance is uniformly 1.
        for (int u = 0; u < userCount; u++) {
            for (int i : trainMatrix.getRowRef(u).indexList()) {
                if (WRMFJudge == 1 || WRMFJudge == 2 || WRMFJudge == 3) {
                    weights.setValue(u, i, 1.0 + Math.log(1.0 + Math.pow(10, weightCoefficient)) * trainMatrix.getValue(u, i));
                } else {//only missing part
                    weights.setValue(u, i, w_init);
                }
                if (WRMFJudge == 3) {
                    //todo 在WRMF基础上，将计算的confidence加在w_ui上
                    //这里是non-zero部分，因此confidence均不为0
                    weights.setValue(u, i, weights.getValue(u, i) * confidences_cache[i]);
                }
            }
        }
    }


    public void setTrain(SparseMatrix trainMatrix) {
        this.trainMatrix = new SparseMatrix(trainMatrix);
        weights = new SparseMatrix(userCount, itemCount);
        for (int u = 0; u < userCount; u++)
            for (int i : this.trainMatrix.getRowRef(u).indexList())
                weights.setValue(u, i, 1);
    }

    // Init SU and SV
    private void initS() {
        SU = U.transpose().mult(U);
        // Init SV as V^T confidences V
        SV = new DenseMatrix(factors, factors);

        //对item的init需要乘上confidence_i
        for (int f = 0; f < factors; f++) {
            for (int k = 0; k <= f; k++) {
                double val = 0;
                for (int i = 0; i < itemCount; i++)
                    val += V.get(i, f) * V.get(i, k) * confidences[i];//c_i*
                SV.set(f, k, val);
                SV.set(k, f, val);
            }
        }
    }

    //remove
    public void setUV(DenseMatrix U, DenseMatrix V) {
        this.U = U.clone();
        this.V = V.clone();
        initS();
    }

    /**
     * for pre-training step
     */
    @Override
    public void buildModel() {
        //System.out.println("Run for FastALS. ");
        double loss_pre = Double.MAX_VALUE;
        for (int iter = 0; iter < maxIter; iter++) {
            Long start = System.currentTimeMillis();
            runOneIteration();

            // Show progress
            if (showProgress)
                showProgress(iter, start, testRatings);
            // Show loss
            if (showLoss)
                loss_pre = showLoss(iter, start, loss_pre);

        } // end for iter

        //todo 训练结束
//        isTrain = false;
    }

    // Run model for one iteration
    public void runOneIteration() {
        // Update user latent vectors
        for (int u = 0; u < userCount; u++) {
            update_user(u);
        }

        // Update item latent vectors
        for (int i = 0; i < itemCount; i++) {
            update_item(i);
        }
    }

    protected void update_user(int u) {
        //获取u在trainMatrix的行中非零项
        ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
        if (itemList.size() == 0) return;    // user has no ratings => new user
//        if (itemList.size() == 0) {
//            if (isTrain && !isNewUser[u]) {
//                isNewUser[u] = true;
//            }
//            return;
//        }
        // prediction cache for the user =>为什么这个要抽出来放在前面，放在下面的foreach in itemList中不行吗？
        for (int i : itemList) {
            prediction_items[i] = predict(u, i);//p_u*q_i=>预测值
            rating_items[i] = trainMatrix.getValue(u, i);//实际值
            w_items[i] = weights.getValue(u, i);//weight
        }

        DenseVector oldVector = U.row(u);//p_u
        for (int f = 0; f < factors; f++) {
            /**
             * p_{u,f}'s numerator = SUM[i in itemList]{[w_{u,i}*r_{u,i}-(w_{u,i}-c_i)*r^_f{u,i}]*q_{i,f}} - SUM[k!=f]{p_{u,k}*s_q{k,f}}
             * p_{u,f}'s denominator = SUM[i in itemList](w_{u,i}-c_i)*sqaure(q_{i,f}) + s_q{f,f} + regUser
             *
             * r^_f{u,i} = r^_{u,i} - p_{u,f}*q_{i,f} = p_u*q_i - p_{u,f}*q_{i,f}
             */

            double numer = 0, denom = 0;//numerator:分子  denominator:分母
            // O(K) complexity for the negative part
            /**
             * numerator : -SUM[k!=f]{p_{u,k}*s_q{k,f}}
             */
            for (int k = 0; k < factors; k++) {
                if (k != f)
                    numer -= U.get(u, k) * SV.get(f, k);
            }
            //numer *= c0;

            // O(Nu) complexity for the positive part
            /**
             * numerator : SUM[i in itemList]{[w_{u,i}*r_{u,i}-(w_{u,i}-c_i)*r^_f{u,i}]*q_{i,f}}
             * denominator : SUM[i in itemList](w_{u,i}-c_i)*sqaure(q_{i,f})
             */
            for (int i : itemList) {
                prediction_items[i] -= U.get(u, f) * V.get(i, f);//r^_f{u,i} = p_u*q_i - p_{u,f}*q_{i,f}
                numer += (w_items[i] * rating_items[i] - (w_items[i] - confidences[i]) * prediction_items[i]) * V.get(i, f);
                denom += (w_items[i] - confidences[i]) * V.get(i, f) * V.get(i, f);
            }
            denom += SV.get(f, f) + regUser;

            // Parameter Update
            U.set(u, f, numer / denom);

            // Update the prediction cache :algorithm line9
            for (int i : itemList)
                prediction_items[i] += U.get(u, f) * V.get(i, f);
        } // end for f

        // Update the SU cache
        for (int f = 0; f < factors; f++) {
            for (int k = 0; k <= f; k++) {
                double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k)
                        + U.get(u, f) * U.get(u, k);
                SU.set(f, k, val);
                SU.set(k, f, val);
            }
        } // end for f
    }

    protected void update_item(int i) {
        ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
        if (userList.size() == 0) return; // item has no ratings.
        // prediction cache for the item
        for (int u : userList) {
            prediction_users[u] = predict(u, i);//p_u*q_i
            rating_users[u] = trainMatrix.getValue(u, i);//实际值
            w_users[u] = weights.getValue(u, i);//w_{u,i}
        }

        DenseVector oldVector = V.row(i);
        for (int f = 0; f < factors; f++) {
            /**
             * q_{i,f}'s numerator = SUM[u in userList]{[w_{u,i}*r_{u,i}-(w_{u,i}-c_i)*r^_f{u,i}]*p_{u,f}} - c_i*SUM[k!=f]{q_{i,k}*s_p{k,f}}
             * q_{i,f}'s denominator = SUM[u in userList](w_{u,i}-c_i)*sqaure(p_{u,f}) + c_i*s_p{f,f} + regItem
             *
             * r^_f{u,i} = r^_{u,i} - p_{u,f}*q_{i,f} = p_u*q_i - p_{u,f}*q_{i,f}
             */
            // O(K) complexity for the c0 part
            double numer = 0, denom = 0;
            for (int k = 0; k < factors; k++) {
                if (k != f)
                    numer -= V.get(i, k) * SU.get(f, k);
            }
            numer *= confidences[i];

            // O(Ni) complexity for the positive ratings part
            for (int u : userList) {
                prediction_users[u] -= U.get(u, f) * V.get(i, f);
                numer += (w_users[u] * rating_users[u] - (w_users[u] - confidences[i]) * prediction_users[u]) * U.get(u, f);
                denom += (w_users[u] - confidences[i]) * U.get(u, f) * U.get(u, f);
            }
            denom += confidences[i] * SU.get(f, f) + regItem;//reg -> regItem

            // Parameter update
            V.set(i, f, numer / denom);
            // Update the prediction cache for the item
            for (int u : userList)
                prediction_users[u] += U.get(u, f) * V.get(i, f);
        } // end for f

        // Update the SV cache
        for (int f = 0; f < factors; f++) {
            for (int k = 0; k <= f; k++) {
                double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k) * confidences[i]
                        + V.get(i, f) * V.get(i, k) * confidences[i];
                SV.set(f, k, val);
                SV.set(k, f, val);
            }
        }
    }

    public double showLoss(int iter, long start, double loss_pre) {
        long start1 = System.currentTimeMillis();
        double loss_cur = loss();
        String symbol = loss_pre >= loss_cur ? "-" : "+";//与之前比loss是否下降
        System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s]\n", iter,
                Printer.printTime(start1 - start), symbol, loss_cur,
                Printer.printTime(System.currentTimeMillis() - start1));
        return loss_cur;
    }

    // Fast way to calculate the loss function
    public double loss() {
//        double L = reg * (U.squaredSum() + V.squaredSum());
        double L = regUser * U.squaredSum() + regItem * V.squaredSum();
        for (int u = 0; u < userCount; u++) {
            double l = 0;
            for (int i : trainMatrix.getRowRef(u).indexList()) {
                double pred = predict(u, i);
                l += weights.getValue(u, i) * Math.pow(trainMatrix.getValue(u, i) - pred, 2);
                l -= confidences[i] * Math.pow(pred, 2);
            }
            l += SV.mult(U.row(u, false)).inner(U.row(u, false));
            L += l;
        }

        return L;
    }

    @Override
    public double predict(int u, int i) {
        return U.row(u, false).inner(V.row(i, false));
    }

    /**
     * todo 根据WRMFJudge不同，对weights和confidencs给予不同赋值
     * 过去仅处理了eals情况：weights.setValue(u, i, w_init)，之前的方法忘记修改了
     */
    @Override
    public void updateModel(int u, int i) {
        trainMatrix.setValue(u, i, 1);
//        weights.setValue(u, i, w_init);
        /**
         * new iteraction => weight至多为1
         */
        if (WRMFJudge == 1 || WRMFJudge == 2 || WRMFJudge == 3) {
            if (weights.getValue(u, i) == 0 || isOnlineWeightUpdate)
                weights.setValue(u, i, 1.0 + Math.log(1.0 + Math.pow(10, weightCoefficient)) * trainMatrix.getValue(u, i));
            //else 暂时用不到
        } else {//WRMFJudge == 0: 无论是否为新interaction
            weights.setValue(u, i, w_init);
        }


        /**
         * 仅当WRMFJudge为0和2时，才起作用，其他设置为1
         */
        if (confidences[i] == 0) {// an new item
//            System.out.println("item[" + i + "] is a new item");
//            if (WRMFJudge == 0 || WRMFJudge == 2)
            confidences[i] = c0 / itemCount;//todo ?? 在推荐过程中怎么能知道总itemCount？？
            // Update the SV cache
            for (int f = 0; f < factors; f++) {
                for (int k = 0; k <= f; k++) {
                    double val = SV.get(f, k) + V.get(i, f) * V.get(i, k) * confidences[i];
                    SV.set(f, k, val);
                    SV.set(k, f, val);
                }
            }
        } else if (confidences_cache != null && confidences_cache[i] == 0) {//WRMF == 3
            confidences_cache[i] = c0 / itemCount;
            weights.setValue(u, i, confidences_cache[i] * weights.getValue(u, i));
        }

        for (int iter = 0; iter < maxIterOnline; iter++) {
            update_user(u);
            update_item(i);
        }
    }

    /**
     * 当修改了WRMFJudge参数时，需要重新计算confidence和weight
     *
     * @param WRMFJudge
     */
    public void setWRMFJudge(int WRMFJudge) {
        if (this.WRMFJudge == WRMFJudge)
            return;
        this.WRMFJudge = WRMFJudge;
        initConfidencesAndWeights();
    }


    public void showParams() {
        System.out.println("WRMFJudge=" + WRMFJudge + ",\tisOnlineWeightUpdate=" + isOnlineWeightUpdate + ",\tfactors=" + factors + ",\tregUser=regItem=" + regUser + ",\talpha=" + alpha + ",\tmaxIter=" + maxIter + ",\tmaxOnlineIter=" + maxIterOnline);
    }

}

