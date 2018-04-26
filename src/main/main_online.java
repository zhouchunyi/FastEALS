package main;

import java.io.IOException;
import java.util.Date;

import data_structure.DenseMatrix;
import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MF_ALS;
import algorithms.MF_CD;
import algorithms.ItemPopularity;
import algorithms.MFbpr;

public class main_online extends main {
    public static void main(String argv[]) throws IOException {
        String dataset_name = "yelp";
        String method = "FastALS";
        int interval = 1000;
        double w0 = 512;
        int factors = 64;
        int maxIter = 50;
        int maxIterOnline = 1;
        double alpha = 0.4;
        String onlineMode = "ui";
        double w_new = 1;

        //从控制台中输入的参数
        if (argv.length > 0) {
            dataset_name = argv[0];
            method = argv[1];
            interval = Integer.parseInt(argv[2]);
            w0 = Double.parseDouble(argv[3]);
            factors = Integer.parseInt(argv[4]);
            maxIter = Integer.parseInt(argv[5]);
            maxIterOnline = Integer.parseInt(argv[6]);
            alpha = Double.parseDouble(argv[7]);
            if (argv.length >= 9) onlineMode = argv[8];
            if (argv.length >= 10) w_new = Double.parseDouble(argv[9]);
        }

        //按照时间先后顺序划分dataset
        ReadRatings_GlobalSplit("data/" + dataset_name + ".rating", 0.1);

        System.out.printf("Online evaluation for %s: factors=%d, maxIter=%d, maxInterOnline=%d, interval=%d, onlineMode(bpr only)=%s\n",
                method, factors, maxIter, maxIterOnline, interval, onlineMode);//todo ??? onlineMode为什么bpr only ???
        System.out.println("====================================================");

        //todo popularity用在哪里了???，注销了也没什么影响呀
        //todo 是MostPopular的推荐算法？？？貌似是的。。。
        ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
        evaluate_model_online(popularity, "Popularity", interval);

        double init_mean = 0;
        double init_stdev = 0.01;
        double reg = 0.01;
        boolean showProgress = false;
        boolean showLoss = false;

        // Remove ALS is not suitable for online learning.
        if (method.equalsIgnoreCase("als")) {
            MF_ALS als = new MF_ALS(trainMatrix, testRatings, topK, threadNum,
                    factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
            als.buildModel();
            als.maxIterOnline = maxIterOnline;
            evaluate_model_online(als, "MF_ALS", interval);
        }

        /**
         * todo ----------------------------------------------------------------------------------------------
         */
        if (method.equalsIgnoreCase("fastals")) {
            MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
                    factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
            fals.w_new = w_new;
            long start = System.currentTimeMillis();
            System.out.println("train start!");
            fals.buildModel();
            System.out.println("train cost:" + Printer.printTime(start - System.currentTimeMillis()));
            fals.maxIterOnline = maxIterOnline;
            evaluate_model_online(fals, "MF_fastALS", interval);
        }
        /**
         * todo ----------------------------------------------------------------------------------------------
         */

        if (method.equalsIgnoreCase("cd")) {
            MF_CD cd = new MF_CD(trainMatrix, testRatings, topK, threadNum,
                    factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
            cd.w_new = w_new;
            cd.buildModel();
            cd.maxIterOnline = maxIterOnline;
            evaluate_model_online(cd, "MF_CD", interval);
        }

        if (method.equalsIgnoreCase("bpr")) {
            MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum,
                    factors, maxIter, 0.01, false, reg, init_mean, init_stdev, 1, showProgress);
            bpr.onlineMode = onlineMode;
            bpr.buildModel();
            bpr.maxIterOnline = maxIterOnline;
            evaluate_model_online(bpr, "BPR", interval);
        }

    } // end main
}
