package main;

import algorithms.ItemPopularity;
import algorithms.MF_fastALS_WRMF;
import algorithms.TopKRecommender;
import utils.Printer;

import java.io.IOException;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.Date;

/**
 * Created by zcy on 2018/5/9.
 */
public class main_XiamiOffline extends main_XiamiBase {
    public static void main(String[] args) throws IOException, ParseException {
        String dataset_folderPath = "D:\\音乐推荐-baseline实验\\xiami-数据处理\\dataBy3KUser_老用户";

        String dataset_trainFilePath = String.valueOf(Paths.get(dataset_folderPath, "TrainData", "trainPair3KUser.txt"));
        String dataset_testFolderPath = String.valueOf(Paths.get(dataset_folderPath, "TestData_online"));
        String method = "FastALS_WRMF";
        int interval = 1000;
        double c0 = 512;//128
        int factors = 64;
        int maxIter = 32;
        int maxIterOnline = 1;
        float alpha = 0.4f;
        String onlineMode = "ui";
        double w_new = 1;

        double init_mean = 0;
        double init_stdev = 0.01;
        double reg = 0.01;
        boolean showProgress = false;
        boolean showLoss = true;

        readAndConvertRecords(dataset_trainFilePath, dataset_testFolderPath, false, 0);

        System.out.printf(new Date() + "\t%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f\n",
                method, showProgress, factors, maxIter, reg, c0, alpha);
        System.out.println("====================================================");

        ItemPopularity popularity = new ItemPopularity(trainMatrix, testRecords, topK, threadNum);
        evaluate_model(popularity, "Popularity");


        if (method.equalsIgnoreCase("fastals_wrmf")) {
            MF_fastALS_WRMF fals_wrmf = new MF_fastALS_WRMF(trainMatrix, testRecords, topK, threadNum, factors, maxIter, c0,
                    alpha, reg, reg, init_mean, init_stdev, showProgress, showLoss);
            fals_wrmf.w_new = w_new;
            evaluate_model(fals_wrmf, "MF_fastALS");
        }
    }

    // Evaluate the model
    public static double[] evaluate_model(TopKRecommender model, String name) {
        long start = System.currentTimeMillis();
        model.buildModel();
        model.evaluate(testRecords);

        double[] res = new double[3];
        res[0] = model.hits.mean();
        res[1] = model.ndcgs.mean();
        res[2] = model.precs.mean();
        System.out.printf("%s\t <hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f [%s]\n",
                name, res[0], res[1], res[2],
                Printer.printTime(System.currentTimeMillis() - start));
        return res;
    }
}
