package data_structure;

/**
 * 内容和Rating是一样的，但是id是String类型的
 * 这里建议把id改为Object类型
 * todo 等到把实验中数组换成map以后再做
 * Created by zcy on 2018/4/25.
 */
public class XiamiRecord {
    public Object userId;
    public Object itemId;
    public int interaction;//通常而言是0/1的
    public long timestamp;//利用long来存储，读取文件时将String转换为long
}
