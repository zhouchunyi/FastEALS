package main;

import algorithms.ItemPopularity;
import algorithms.MF_fastALS_WRMF;
import algorithms.TopKRecommender;
import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.SparseVector;
import utils.IdIndexConvertor;
import utils.Printer;

import java.io.*;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;


/**
 * 用于执行xiami数据的main
 * todo test中的user和item都是train中出现过的
 * Created by zcy on 2018/4/25.
 */
public class main_XiamiBase {
    /***********************Data Structures*********************************/
    /**
     * Rating matrix for training.
     */
    public static SparseMatrix trainMatrix;

    /**
     * Test records (sorted by time).
     * 仍利用Rating数据结构，但在Xiami中为record
     */
    public static ArrayList<Rating> testRecords;//Rating: <userId, itemId, score, timestamp>
    /**
     * newUser indices list in test ratings
     */
    public static ArrayList<Integer> newUserIdxList;//size = newUserCount
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
    //train+test的数量
    public static int userCount, itemCount;
    //用户数量分布
    public static int trainUserCount, testUserCount;
    //用于从test中挑选newUser的随机数
    public static int randomSeed = 1;
    private static Random newUserRandom;

    public static int topK = 100;
    public static int threadNum = 10;


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

        File trainTripleFile = new File(trainTripleFilePath);
        File testFolder = new File(testFolderPath);
        if ((!trainTripleFile.exists()) || (!testFolder.exists())) {
            System.out.println("train or test Folder does NOT exist!");
            return;
        }

        long start;

        BufferedReader reader;
        String line;
        String[] splitRes;
        int userIdx, itemIdx;
        //build train
        System.out.println(new Date() + "\tbuilding train-dataset start.");
        /**
         * 由于train部分和timestamp先后没有关联，因此直接输入三元组
         */
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainTripleFile)));
        ArrayList<Rating> trainList = new ArrayList<>();//temp var
//        start = System.currentTimeMillis();
        while ((line = reader.readLine()) != null) {
            //userId \t itemId \t freq
            splitRes = line.split(" ");//todo trainPair文件用空格分
            userIdx = convertor.putUser(splitRes[0]);
            itemIdx = convertor.putItem(splitRes[1]);
            Rating rating = new Rating(userIdx, itemIdx, Float.parseFloat(splitRes[2]), 0L);
            trainList.add(rating);
        }
        trainUserCount = convertor.getUserNum();
        System.out.println(new Date() + "\tbuilding train-dataset ends.");

        String userId;
        long timestamp;
        /**
         *  记录所有出现在test中的用户index
         *  用于随机挑选newUser
         */
        testRecords = new ArrayList<>();
        //todo
        ArrayList<Integer> testUserIdxList = new ArrayList<>();
        //build test
        System.out.println(new Date() + "\tbuilding test-dataset start.");
        newUserIdxList = new ArrayList<>();
        /**
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
                testRecords.add(rating);
            }
        }
        System.out.println(new Date() + "\tbuilding test-dataset ends. Reading testFile costs:\t" + Printer.printTime(System.currentTimeMillis() - start));

        //在此处构建trainMatrix为了防止test中有train中没有的user
        //此方法为静态matrix，因此train中仍为new user保留行
        userCount = convertor.getUserNum();
        itemCount = convertor.getItemNum();
        trainMatrix = new SparseMatrix(userCount, itemCount);
//        System.out.println("after reading train + test\t userCount = " + userCount + ",\titemCount = " + itemCount);

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
        System.out.println(new Date() + "\tsorting test-records starts.");
        Collections.sort(testRecords, c);
        System.out.println(new Date() + "\tsorting test-records ends. Sorting cost:" + Printer.printTime(System.currentTimeMillis() - start));

        /**
         * *************构建新用户****************
         * 保证newUser在testUser中占比为newUserRatio
         */
        if (needNewUser) {
            int testUserNum = testUserIdxList.size();

            DecimalFormat df = new DecimalFormat("#####0");//四舍五入
            int goalNewUserNum = Integer.parseInt(df.format(testUserNum * newUserRatio));
//            int goalNewUserNum = (int) (testUserNum * newUserRatio);

            System.out.println("Need NewUser, newUserRatio = " + newUserRatio);
            System.out.println("Now, #testUser = " + testUserNum + "#currentNewUser = " + newUserIdxList.size() + ", the goal of #NewUser = " + goalNewUserNum);
            System.out.println("Before choosing NewUser from test, #Records in Train\t" + (int) trainMatrix.sum());

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
//                System.out.println("random choose:" + chosenUIdx);
                if (newUserIdxList.contains(chosenUIdx))//这个user已经是new的
                    continue;
                trainMatrix.setRowVector(chosenUIdx, zeroVector);
                newUserIdxList.add(chosenUIdx);
                goalNewUserNum--;
            }
        }

        //之前的计算有误
        trainUserCount = userCount - newUserIdxList.size();
        testUserCount = testUserIdxList.size();
        // Print some basic statistics of the dataset.
        System.out.println("Data:\ttrainFile\t" + trainTripleFilePath + "\ttestFile\t" + testFolderPath);
        System.out.println("#trainUserNum\t" + trainUserCount + " ,#testUserNum\t" + testUserCount + ", #newUser_in_test\t" + newUserIdxList.size());
        System.out.println("#TotalUsers\t" + userCount);
        System.out.println("#TotalItems\t" + itemCount);
        System.out.println("#TotalTrainPlayCount\t" + (int) trainMatrix.sum());
        System.out.println("#TestRecord\t" + testRecords.size());
    }
}
