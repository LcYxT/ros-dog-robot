# 106Dogrobot

106專題 智慧導盲機器人，再見了可魯

利用聲音辨識，影像辨識等等的功能，模擬出和原本導盲犬甚至於超越導盲犬的功能，做到導盲犬所做不到的辨識紅綠燈。

## 系統環境：
Ｕbuntu 16.04

Python 2.7.12

Snowboy 1.1.1

Opencv 3.4.0

## 環境安裝：
安裝ROS
> http://wiki.ros.org/kinetic/Installation

創建catkin workspace
> http://wiki.ros.org/catkin/Tutorials/create_a_workspace

下載turtlebot_navigation的catkin package
> https://github.com/turtlebot/turtlebot_apps/tree/indigo/turtlebot_navigation

把turtlebot_navigation放到catkin/src裡面
開終端機打指令：
> ~$ cd ~/catkin_ws/

> ~$ catkin_make

最後下載106Dogrobot解壓縮放到catkin/src裡面

## 使用方法：
### 模式一：
先bringup turtlebot

開一個終端機打指令：
> ~$ roslaunch turtlebot_bringup minimal.launch


打開終端機**到放置106Dogrobot的檔案的地方**打指令：
> ~$ python dogrobot.py

然後輸入１按enter，即可開始用聲音控制機器人直走左轉右轉等等。

### 模式二：
先bringup turtlebot

開一個終端機打指令：
> ~$ roslaunch turtlebot_bringup minimal.launch

再開一個終端機打指令：
> ~$ roslaunch turtlebot_navigation amcl_demo.launch map_file:=/導航地圖放置的路徑/commm.yaml

打開終端機**到放置106Dogrobot的檔案的地方**打指令：
> ~$ python dogrobot.py

然後輸入２按enter，說出自己之新增的地方的地名，即可開始導航。

### 新增Snowboy詞語方法：
在snowboy的dashboard新增詞語
> https://snowboy.kitt.ai/dashboard

新增完成後會得到一個.pmdl檔，在程式碼中使用它即可。

### 注意事項：
開啟dogrobot.py的終端機在每一次執行完停止之後都需要關掉重開。


