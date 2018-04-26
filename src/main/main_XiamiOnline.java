package main;

import algorithms.ItemPopularity;
import algorithms.MF_fastALS;
import algorithms.TopKRecommender;
import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.SparseVector;
import utils.IdIndexConvertor;
import utils.Printer;

import java.io.*;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;


/**
 * 用于执行xiami数据的main
 * todo test中的user和item都是train中出现过的
 * Created by zcy on 2018/4/25.
 */
public abstract class main_XiamiOnline {
    /***********************Data Structures*********************************/
    /**
     * Rating matrix for training.
     */
    public static SparseMatrix trainMatrix;

    /**
     * Test ratings (sorted by time).
     */
    public static ArrayList<Rating> testRatings;//Rating: <userId, itemId, score, timestamp>
    /**
     * newUser indices list in test ratings
     */
    public static ArrayList<Integer> newUserIdxList;
    /**
     * 用于存储xiami中String类型的id和int类型的index的转换
     * index对应了Matrix的位置
     */
    private static IdIndexConvertor convertor = new IdIndexConvertor();

    /*************************Properties*********************************/
    /**
     * xiami数据时间戳格式
     */
    public final static SimpleDateFormat STD_DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd hh:mm");

    public static int topK = 100;
    public static int threadNum = 10;
    //train+test的数量
    public static int userCount;
    public static int itemCount;
    //train的数量
    public static int trainUserCount;
    //用于从test中挑选newUser的随机数
    public static int randomSeed = 1;
    private static Random newUserRandom;


    /**
     * 对已经划分好的train和test构建数据结构
     * 文件名为<userId>.txt
     * test每行内容为: timestamp<\t>itemId
     * <p>
     * 结果：
     * 1、构建完毕trainMatrix和testRatings
     * 2、构建完毕id和index的相互映射
     *
     * @param trainTripleFilePath 已经转换好的trainPair文件
     * @param testFolderPath      所有测试用户的播放记录文件夹
     * @param needNewUser         是否需要将test中部分用户设置为newUser（可能存在test中所有用户是老用户或新用户比例极低）
     * @param newUserRatio        test中newUser占test中User的比例，仅在needNewUser为true时有作用
     */
    public static void readAndConvertRecords(String trainTripleFilePath, String testFolderPath, boolean needNewUser, double newUserRatio) throws IOException, ParseException {
        userCount = itemCount = 0;
        trainUserCount = 0;

        File trainTripleFile = new File(trainTripleFilePath);
        File testFolder = new File(testFolderPath);
        if ((!trainTripleFile.exists()) || (!testFolder.exists())) {
            System.out.println("train or test Folder does NOT exist!");
            return;
        }

        long start, functionStart = System.currentTimeMillis();

        BufferedReader reader;
        String line;
        String[] splitRes;
        int userIdx, itemIdx;
        //build train
        System.out.println("Building train start!");
        //由于train部分和timestamp先后没有关联，因此直接输入三元组
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainTripleFile)));
        ArrayList<Rating> trainList = new ArrayList<>();//temp var
        start = System.currentTimeMillis();
        while ((line = reader.readLine()) != null) {
            //userId \t itemId \t freq
            splitRes = line.split(" ");//todo trainPair文件用空格分
            userIdx = convertor.putUser(splitRes[0]);
            itemIdx = convertor.putItem(splitRes[1]);
            Rating rating = new Rating(userIdx, itemIdx, Float.parseFloat(splitRes[2]), 0L);
            trainList.add(rating);
        }
        trainUserCount = convertor.getUserNum();
        System.out.println("Reading trainFile costs :" + Printer.printTime(System.currentTimeMillis() - start));

        String userId;
        long timestamp;
        /**
         *  记录所有出现在test中的用户index
         *  用于随机挑选newUser
         */
        testRatings = new ArrayList<>();
        //todo
        ArrayList<Integer> testUserIdxList = new ArrayList<>();
        //build test
        System.out.println("Building test start!");
        newUserIdxList = new ArrayList<>();
        /*
        testFolder的名字为<userId>.txt，因此test中无重复userId
         */
        start = System.currentTimeMillis();
        for (File testUserFile : testFolder.listFiles()) {
            userId = testUserFile.getName().substring(0, testUserFile.getName().length() - 4);//get userId

            if (!convertor.contrainsUserId(userId)) {
                userIdx = convertor.putUser(userId);
                newUserIdxList.add(userIdx);
            } else
                userIdx = convertor.putUser(userId);

            testUserIdxList.add(userIdx);

            reader = new BufferedReader(new InputStreamReader(new FileInputStream(testUserFile)));
            //timestamp \t itemId
            while ((line = reader.readLine()) != null) {
                splitRes = line.split("\t");
                timestamp = STD_DATE_FORMAT.parse(splitRes[0]).getTime();
                itemIdx = convertor.putItem(splitRes[1]);
                Rating rating = new Rating(userIdx, itemIdx, 1, timestamp);
                testRatings.add(rating);
            }
        }
        System.out.println("Reading testFile costs:\t" + Printer.printTime(System.currentTimeMillis() - start));

        //在此处构建trainMatrix为了防止test中有train中没有的user
        //此方法为静态matrix，因此train中仍为new user保留行
        userCount = convertor.getUserNum();
        itemCount = convertor.getItemNum();
        trainMatrix = new SparseMatrix(userCount, itemCount);

        for (Rating rating : trainList) {
            trainMatrix.setValue(rating.userId, rating.itemId, rating.score);
        }
        trainList.clear();//释放空间

        //将testRating按照timestamp排序
        Comparator<Rating> c = new Comparator<Rating>() {
            public int compare(Rating o1, Rating o2) {
                if (o1.timestamp - o2.timestamp > 0) return 1;
                else if (o1.timestamp - o2.timestamp < 0) return -1;
                else return 0;
            }
        };
        System.out.println("Sorting test ratings!");

        Collections.sort(testRatings, c);
        System.out.println("Sorting cost:" + Printer.printTime(System.currentTimeMillis() - start));

        if (needNewUser) {
            int testUserNum = testUserIdxList.size();

            DecimalFormat df = new DecimalFormat("#####0");//四舍五入
            int goalNewUserNum = Integer.parseInt(df.format(testUserNum * newUserRatio));
//            int goalNewUserNum = (int) (testUserNum * newUserRatio);

            System.out.println("Need NewUser, newUserRatio = " + newUserRatio);
            System.out.println("Now, testUserNum = " + testUserNum + ", the goal of NewUserNum = " + goalNewUserNum);
            System.out.println("Before choosing NewUser from test, #Ratings in Train\t" + (int) trainMatrix.sum());

            goalNewUserNum -= newUserIdxList.size();//当前我们需要生成的新用户
            /**
             * todo 从test用户中挑选一些user作为新用户，即将其train中记录删除
             */
            newUserRandom = new Random(randomSeed);
            /**
             * SparseMatrix.setRowVector是将newVector中每个元素依次setVaule，因此不用担心浅拷贝
             */
            SparseVector zeroVector = new SparseVector(itemCount);
            int chosenUIdx;
            while (goalNewUserNum > 0) {
                chosenUIdx = testUserIdxList.get(newUserRandom.nextInt(testUserNum));//random.nextInt(bound): [0,bound)
                System.out.println("random choose:" + chosenUIdx);
                if (newUserIdxList.contains(chosenUIdx))//这个user已经是new的
                    continue;
                trainMatrix.setRowVector(chosenUIdx, zeroVector);
                newUserIdxList.add(chosenUIdx);
                trainUserCount++;
                goalNewUserNum--;
            }
        }


        // Print some basic statistics of the dataset.
        System.out.println("Data:\ttrainFile\t" + trainTripleFilePath + "\ttestFile\t" + testFolderPath);
        System.out.println("#trainUserNum\t" + trainUserCount + " ,#testUserNum\t" + testUserIdxList.size() + ", #newUser_in_test\t" + newUserIdxList.size());
        System.out.println("#Users\t" + userCount);
        System.out.println("#Items\t" + itemCount);
        System.out.printf("#Ratings\t %d (train), %d(test)\n", (int) trainMatrix.sum(), testRatings.size());
    }


    public static void main(String[] argv) throws IOException, ParseException {
        String dataset_trainFilePath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\最近听歌记录_排序\\FixedSplitResults_filtered\\splitBy2018-03-06_onlyrecent\\trainPair_测试.txt";
        String dataset_testFolderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\最近听歌记录_排序\\FixedSplitResults_filtered\\splitBy2018-03-06_onlyrecent\\test_测试";


        String method = "FastALS";
        int interval = 100;
        double w0 = 512;
        int factors = 64;
        int maxIter = 50;
        int maxIterOnline = 1;
        double alpha = 0.4;
        String onlineMode = "ui";
        double w_new = 1;

        //从控制台中输入的参数
        if (argv.length > 0) {
            dataset_trainFilePath = argv[0];
            dataset_testFolderPath = argv[1];
            method = argv[2];
            interval = Integer.parseInt(argv[3]);
            w0 = Double.parseDouble(argv[4]);
            factors = Integer.parseInt(argv[5]);
            maxIter = Integer.parseInt(argv[6]);
            maxIterOnline = Integer.parseInt(argv[7]);
            alpha = Double.parseDouble(argv[8]);
            if (argv.length >= 10) onlineMode = argv[9];
            if (argv.length >= 11) w_new = Double.parseDouble(argv[10]);
        }


//        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, false, 0);
        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, true, 0.1);

        ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
        evaluate_model_online(popularity, "Popularity", interval);//import from main.java

        double init_mean = 0;
        double init_stdev = 0.01;
        double reg = 0.01;
        boolean showProgress = false;
        boolean showLoss = false;

        if (method.equalsIgnoreCase("fastals")) {
            MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
                    factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
            fals.w_new = w_new;
            long start = System.currentTimeMillis();
            System.out.println("train start!");
            fals.buildModel();
            System.out.println("train costs\t" + Printer.printTime(start - System.currentTimeMillis()));
            fals.maxIterOnline = maxIterOnline;
            evaluate_model_online(fals, "MF_fastALS", interval);
        }
    }


    private static void evaluate_model_online(TopKRecommender model, String name, int interval) {
        long start = System.currentTimeMillis();
        model.evaluateOnline(testRatings, interval);
        System.out.printf("%s\t <hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f [%s]\n",
                name, model.hits.mean(), model.ndcgs.mean(), model.precs.mean(),
                Printer.printTime(System.currentTimeMillis() - start));
    }
}
