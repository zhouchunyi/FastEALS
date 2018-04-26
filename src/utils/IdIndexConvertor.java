package utils;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by zcy on 2018/4/25.
 */
public class IdIndexConvertor {
    private int userNum;
    private int itemNum;
    /**
     * id Convertor
     * index是用于Matrix的索引
     */
    public Map<String, Integer> userIdIdxMap;//<userId,userIndex>
    public Map<Integer, String> userIdxIdMap;//<userIndex,userId>
    public Map<String, Integer> itemIdIdxMap;//<itemId,itemIndex>
    public Map<Integer, String> itemIdxIdMap;//<itemIndex,itemId>

    /**
     * init
     */
    public IdIndexConvertor() {
        userNum = 0;
        itemNum = 0;

        userIdIdxMap = new HashMap<>();
        userIdxIdMap = new HashMap<>();
        itemIdIdxMap = new HashMap<>();
        itemIdxIdMap = new HashMap<>();
    }

    public int putUser(String userId) {
        if (userIdIdxMap.containsKey(userId)) {
            return userIdIdxMap.get(userId);
        } else {
            userIdIdxMap.put(userId, userNum);
            userIdxIdMap.put(userNum, userId);
            return userNum++;
        }
    }

    public int putItem(String itemId) {
        if (itemIdIdxMap.containsKey(itemId)) {
            return itemIdIdxMap.get(itemId);
        } else {
            itemIdIdxMap.put(itemId, itemNum);
            itemIdxIdMap.put(itemNum, itemId);
            return itemNum++;
        }
    }

    public boolean contrainsUserId(String userId) {
        return userIdIdxMap.containsKey(userId);
    }

    public boolean contrainsItemId(String itemId) {
        return itemIdIdxMap.containsKey(itemId);
    }

    /**
     * Get Functions
     * 如果key不存在，则返回null
     */

    public String getUserIdByIndex(int userIdx) {
        return userIdxIdMap.get(userIdx);
    }

    public String getItemIdByIndex(int itemIdx) {
        return itemIdxIdMap.get(itemIdx);
    }

    public int getUserIndexById(String userId) {
        return userIdIdxMap.get(userId);
    }

    public int getItemIndexById(String itemId) {
        return itemIdIdxMap.get(itemId);
    }

    public int getUserNum() {
        return userNum;
    }

    public int getItemNum() {
        return itemNum;
    }
}
