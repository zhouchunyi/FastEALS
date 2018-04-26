import utils.IdIndexConvertor;

import java.io.*;
import java.text.DecimalFormat;

/**
 * Created by zcy on 2018/4/25.
 */
public class Test {
    public static void main(String[] args) throws IOException {
//        File trainPairFile = new File("D:\\音乐推荐-baseline实验\\xiami-数据处理\\最近听歌记录_排序\\FixedSplitResults_filtered\\splitBy2018-03-06_onlyrecent\\trainPair_0306.txt");
//        String line;
//        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainPairFile)));
//        for (int i = 0; i < 10000; i++) {
//            line = reader.readLine();
//            System.out.println(line);
//        }

//        IdIndexConvertor convertor = new IdIndexConvertor();
//        System.out.println(convertor.putUser("hello"));
//        System.out.println(convertor.putUser("hello"));
//        System.out.println(convertor.getItemIdByIndex(2));

        int testUserNum = 166666;
        double newUserRatio = 0.1;
        DecimalFormat df = new DecimalFormat("##0");
//        int goalNewUserNum = df.format(testUserNum * newUserRatio);
        System.out.println(df.format(testUserNum * newUserRatio));
    }
}
