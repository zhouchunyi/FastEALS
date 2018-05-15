package main;

import algorithms.ItemPopularity;
import algorithms.MF_fastALS_WRMF;
import algorithms.MF_fastALS_WRMFonline;
import algorithms.TopKRecommender;
import utils.Printer;

import java.io.IOException;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.Date;

/**
 * Created by zcy on 2018/5/9.
 */
public class main_XiamiOnline extends main_XiamiBase {
    public static void main(String[] args) throws IOException, ParseException {
//        xnh_tuningMain();
//        banditLikeMain();
        xnh_onlineMain();
    }

    public static void xnh_onlineMain() throws IOException, ParseException {
        String dataset_folderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\dataBy3KUser_老用户";

        String dataset_trainFilePath = String.valueOf(Paths.get(dataset_folderPath, "TrainData", "trainPair3KUser.txt"));
        String dataset_testFolderPath = String.valueOf(Paths.get(dataset_folderPath, "TestData_online"));
//        String dataset_folderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\郑光煜实验数据\\dataBy100users_FastEALS";
//
//        String dataset_trainFilePath = String.valueOf(Paths.get(dataset_folderPath, "100UserTrain.txt"));
//        String dataset_testFolderPath = String.valueOf(Paths.get(dataset_folderPath, "test_online"));

        int interval = 10000;
        double c0 = 512;//128
        int factors = 64;
        int maxIter = 32;
        int maxIterOnline = 1;
        float alpha = 0.4f;
        String onlineMode = "ui";
        double w_new = 1;

//        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, true, 0.1);
        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, false, 0);//不人为新增newUser
//
//        ItemPopularity popularity = new ItemPopularity(trainMatrix, testRecords, topK, threadNum);
//        evaluate_model_online(popularity, "Popularity", interval);//import from main.java

        double init_mean = 0;
        double init_stdev = 0.01;
        double reg = 0.1;
        boolean showProgress = false;
        boolean showLoss = false;

        int[] WRMFJudge = {3, 1, 2, 0};
//        int[] WRMFJudge = {3};
        for (int judge : WRMFJudge) {
            MF_fastALS_WRMF fals_wrmf = new MF_fastALS_WRMF(trainMatrix, testRecords, judge, topK, threadNum, factors, maxIter, c0,
                    alpha, reg, reg, init_mean, init_stdev, showProgress, showLoss);
            fals_wrmf.w_init = w_new;
            fals_wrmf.showParams();
            long start = System.currentTimeMillis();
            System.out.println(new Date() + "\tfastals_wrmf train starts.");
            fals_wrmf.buildModel();
            System.out.println(new Date() + "\tfastals_wrmf train ends. Train costs\t" + Printer.printTime(start - System.currentTimeMillis()));
            fals_wrmf.maxIterOnline = maxIterOnline;
            evaluate_model_online(fals_wrmf, fals_wrmf.method_name, interval);
        }
    }

    private static void evaluate_model_online(TopKRecommender model, String name, int interval) {
        long start = System.currentTimeMillis();
        model.evaluateOnline(testRecords, interval);
        System.out.printf("%s\t <hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f [%s]\n",
                name, model.hits.mean(), model.ndcgs.mean(), model.precs.mean(),
                Printer.printTime(System.currentTimeMillis() - start));
    }

    public static void xnh_tuningMain() throws IOException, ParseException {
        String dataset_folderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\dataBy3KUser_老用户";

        String dataset_trainFilePath = String.valueOf(Paths.get(dataset_folderPath, "TrainData", "trainPair3KUser.txt"));
        String dataset_testFolderPath = String.valueOf(Paths.get(dataset_folderPath, "TestData_online"));

        String method = "FastALS_WRMF";
        int interval = 10000;
//        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, true, 0.1);
        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, false, 0);//不人为新增newUser

        double init_mean = 0;
        double init_stdev = 0.01;
        boolean showProgress = false;
        boolean showLoss = false;
        String onlineMode = "ui";
        //todo 这个参数还没有调节
        double c0 = 512;
        double w_new = 1;

        //tuning
        int[] factors = {64, 32, 16};
        double[] regs = {0.1, 0.05, 0.2, 0.02, 0.5, 0.01};
        float[] alphas = {0.4f, 0.5f, 1f, 0.2f, 0.8f};
        int[] iters = {32, 16, 8, 64};
        int[] onlineIters = {1, 2, 5};

        int WRMFJudge = 1;

        for (int factor : factors)
            for (int iter : iters)
                for (double reg : regs)
                    for (float alpha : alphas)
                        for (int onlineIter : onlineIters) {
                            MF_fastALS_WRMF fals_wrmf = new MF_fastALS_WRMF(trainMatrix, testRecords, WRMFJudge, topK, threadNum, factor, iter, c0,
                                    alpha, reg, reg, init_mean, init_stdev, showProgress, showLoss);
                            fals_wrmf.maxIterOnline = onlineIter;
                            fals_wrmf.w_init = w_new;
                            fals_wrmf.showParams();

                            long start = System.currentTimeMillis();
                            System.out.println(new Date() + "\tfastals_wrmf train starts.");
                            fals_wrmf.buildModel();
                            System.out.println(new Date() + "\tfastals_wrmf train ends. Train costs\t" + Printer.printTime(start - System.currentTimeMillis()));
//                            fals_wrmf.maxIterOnline = onlineIter;
                            evaluate_model_online(fals_wrmf, fals_wrmf.method_name, interval);
                            System.out.println("===============================================================\n");
                        }

    }

    public static void banditLikeMain() throws IOException, ParseException {
//        String dataset_folderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\dataBy3KUser_老用户";
//
//        String dataset_trainFilePath = String.valueOf(Paths.get(dataset_folderPath, "TrainData", "trainPair3KUser.txt"));
//        String dataset_testFolderPath = String.valueOf(Paths.get(dataset_folderPath, "TestData_online"));
        String dataset_folderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\郑光煜实验数据\\dataBy100users_FastEALS";

        String dataset_trainFilePath = String.valueOf(Paths.get(dataset_folderPath, "100UserTrain.txt"));
        String dataset_testFolderPath = String.valueOf(Paths.get(dataset_folderPath, "test_online"));

        int interval = 10000;
        double c0 = 512;//128
        int factors = 64;
        int maxIter = 5;
        int maxIterOnline = 1;
        float alpha = 0.4f;
        String onlineMode = "ui";
        double w_new = 1;
        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, false, 0);//不人为新增newUser
        double init_mean = 0;
        double init_stdev = 0.01;
        double reg = 0.01;
        boolean showProgress = false;
        boolean showLoss = false;
        int WRMFJudge = 1;
        MF_fastALS_WRMFonline fals_wrmf = new MF_fastALS_WRMFonline(trainMatrix, testRecords, WRMFJudge, topK, factors, maxIter, c0,
                alpha, reg, reg, init_mean, init_stdev, showProgress, showLoss);
        fals_wrmf.w_init = w_new;
        fals_wrmf.showParams();
        long start = System.currentTimeMillis();
        System.out.println(new Date() + "\tfastals_wrmf_online train starts.");
        fals_wrmf.buildModel();
        System.out.println(new Date() + "\tfastals_wrmf_online train ends. Train costs\t" + Printer.printTime(start - System.currentTimeMillis()));
        fals_wrmf.maxIterOnline = maxIterOnline;

        long evalStart = System.currentTimeMillis();
        fals_wrmf.evaluateOnline(testRecords, interval);
        System.out.printf("%s\t <hit>:\t %.4f\t [%s]\n",
                fals_wrmf.method_name, fals_wrmf.hits.mean(),
                Printer.printTime(System.currentTimeMillis() - evalStart));

    }
}
