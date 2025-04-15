<template>
<div>
<!--  第一行-->
  <div style="margin-top: 40px;margin-left: 20px;display: flex">
    <div style="width: 50%">
      <div id="line" style="width: 100%;height: 400px"></div>
    </div>
    <div style="width: 50%;margin-left: 20px">
      <div id="bar" style="width: 100%;height: 400px"></div>
    </div>
  </div>
<!--第二行-->
  <div style="margin-top: 10px;margin-left: 20px;display: flex;">
    <div style="width: 50%;margin-left: 20px">
      <div id="line1" style="width: 100%;height: 400px"></div>
    </div>
    <div style="width: 50%">
      <div id="black" style="width: 100%;height: 400px"></div>
    </div>
  </div>

</div>



</template>

<script>
import * as echarts from 'echarts';
export default {
  name: "Index",
  mounted() {
    this.initLine();
    this.initBar();
    this.initLine1();
    this.initBlack();
  },
  data(){
    return{
      // lineData: null,

      seriesData: [
        { value: 981, name: '流量异常' },
        { value: 735, name: '时序异常' },
        { value: 484, name: '端口异常' },
        { value: 300, name: '标志位异常' }
      ],
      ruleMatchData: [
        0.672,
        0.641,
        0.543,
        0.721
      ],
      isolationForestArray: [
        0.786,
        0.784,
        0.788,
        0.864
      ],
      cnnArray: [
        0.712,
        0.754,
        0.615,
        0.823
      ]
      // 其他可能需要缓存的数据
    }
  },
  methods: {
    initLine(){
      let chartDom = document.getElementById('line');
      let myChart = echarts.init(chartDom);
      let option;


      option = {
        title: {
          text: '24小时网络流量折线图',
          subtext: '2024-11-09'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross'
          }
        },
        toolbox: {
          show: true,
          feature: {
            saveAsImage: {}
          }
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          // prettier-ignore
          data: ['00:00', '01:15', '02:30', '03:45', '05:00', '06:15', '07:30', '08:45', '10:00', '11:15', '12:30', '13:45', '15:00', '16:15', '17:30', '18:45', '20:00', '21:15', '22:30', '23:45']
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: '{value} Byte'
          },
          axisPointer: {
            snap: true
          }
        },
        visualMap: {
          show: false,
          dimension: 0,
          pieces: [
            {
              lte: 6,
              color: 'green'
            },
            {
              gt: 6,
              lte: 10,
              color: 'red'
            },
            {
              gt: 10,
              lte: 14,
              color: 'green'
            },
            {
              gt: 14,
              lte: 18,
              color: 'red'
            },
            {
              gt: 18,
              color: 'green'
            }
          ]
        },
        series: [
          {
            name: '实时流量',
            type: 'line',
            smooth: true,
            // prettier-ignore
            data: [300, 280, 250, 260, 270, 380, 520, 575, 540, 525, 500, 460, 400, 500, 600, 750, 800, 700, 600, 400],
            markArea: {
              itemStyle: {
                color: 'rgba(255, 173, 177, 0.4)'
              },
              data: [
                [
                  {
                    name: 'Morning Peak',
                    xAxis: '07:30'
                  },
                  {
                    xAxis: '12:30'
                  }
                ],
                [
                  {
                    name: 'Evening Peak',
                    xAxis: '17:30'
                  },
                  {
                    xAxis: '22:30'
                  }
                ]
              ]
            }
          }
        ]
      };

      option && myChart.setOption(option);
    },
    initBar() {
      let chartDom = document.getElementById('bar');
      let myChart = echarts.init(chartDom);
      let option;



      // 调用后端接口获取规则匹配的不同异常类型数量
      // fetch('/api/ruleMatch_c')
      //     .then(response => response.json())
      //     .then(data => {
      //       let seriesData = [
      //         { value: data.size, name: '流量异常' },
      //         { value: data.time, name: '时序异常' },
      //         { value: data.port, name: '端口异常' },
      //         { value: data.flag, name: '标志位异常' }
      //       ];
      //
      //       option = {
      //         title: {
      //           text: '异常数据分布图',
      //           left: 'center'
      //         },
      //         tooltip: {
      //           trigger: 'item'
      //         },
      //         legend: {
      //           orient: 'vertical',
      //           left: 'left'
      //         },
      //         series: [
      //           {
      //             name: '异常类型',
      //             type: 'pie',
      //             radius: '50%',
      //             // data: seriesData,
      //             data: seriesData,
      //             emphasis: {
      //               itemStyle: {
      //                 shadowBlur: 10,
      //                 shadowOffsetX: 0,
      //                 shadowColor: 'rgba(0, 0, 0, 0.5)'
      //               }
      //             }
      //           }
      //         ]
      //       };
      //
      //       option && myChart.setOption(option);
      //     });
      option = {
        title: {
          text: '异常数据分布图',
          left: 'center'
        },
        tooltip: {
          trigger: 'item'
        },
        legend: {
          orient: 'vertical',
          left: 'left'
        },
        series: [
          {
            name: '异常类型',
            type: 'pie',
            radius: '50%',
            data: this.seriesData,
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ]
      };

      option && myChart.setOption(option);
    },
    initLine1() {
      let chartDom = document.getElementById('line1');
      let myChart = echarts.init(chartDom);
      let option;



      // 调用基于规则匹配的异常检测接口
      // fetch('/api/ruleMatch_v')
      //     .then(response => response.json())
      //     .then(data => {
      //       let ruleMatchData = [
      //         data.f1,
      //         data.precision,
      //         data.recall,
      //         data.accuracy
      //       ];
      //       // 调用基于无监督的孤立森林接口
      //       return fetch('/api/isolationForest_v')
      //           .then(response => response.json())
      //           .then(isolationForestData => {
      //             let isolationForestArray = [
      //               isolationForestData.f1,
      //               isolationForestData.precision,
      //               isolationForestData.recall,
      //               isolationForestData.accuracy
      //             ];
      //             // 调用基于CNN的接口
      //             return fetch('/api/cnn_v')
      //                 .then(response => response.json())
      //                 .then(cnnData => {
      //                   let cnnArray = [
      //                     cnnData.f1,
      //                     cnnData.precision,
      //                     cnnData.recall,
      //                     cnnData.accuracy
      //                   ];
      //
      //                   option = {
      //                     title: {
      //                       text: '模型评估'
      //                     },
      //                     tooltip: {
      //                       trigger: 'axis',
      //                       axisPointer: {
      //                         type: 'shadow'
      //                       }
      //                     },
      //                     legend: {},
      //                     grid: {
      //                       left: '3%',
      //                       right: '4%',
      //                       bottom: '3%',
      //                       containLabel: true
      //                     },
      //                     xAxis: {
      //                       type: 'value',
      //                       boundaryGap: [0, 0.01]
      //                     },
      //                     yAxis: {
      //                       type: 'category',
      //                       data: ['F1值', '精确率', '召回率', '准确率']
      //                     },
      //                     series: [
      //                       {
      //                         name: '规则匹配',
      //                         type: 'bar',
      //                         data: ruleMatchData
      //                       },
      //                       {
      //                         name: '孤立森林',
      //                         type: 'bar',
      //                         data: isolationForestArray
      //                       },
      //                       {
      //                         name: 'CNN',
      //                         type: 'bar',
      //                         data: cnnArray
      //                       }
      //                     ]
      //                   };
      //
      //                   option && myChart.setOption(option);
      //                 });
      //           });
      //     });
      option = {
        title: {
          text: '模型评估'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          }
        },
        legend: {},
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'value',
          boundaryGap: [0, 0.01]
        },
        yAxis: {
          type: 'category',
          data: ['F1值', '精确率', '召回率', '准确率']
        },
        series: [
          {
            name: '规则匹配',
            type: 'bar',
            data: this.ruleMatchData
          },
          {
            name: '孤立森林',
            type: 'bar',
            data: this.isolationForestArray
          },
          {
            name: 'CNN',
            type: 'bar',
            data: this.cnnArray
          }
        ]
      };

      option && myChart.setOption(option);
    },
    initBlack(){
      let chartDom = document.getElementById('black');
      let myChart = echarts.init(chartDom);
      let option;



      option = {
        title: {
          text: '黑白名单分布情况'
        },
        legend: {
          data: ['黑名单', '白名单']
        },
        radar: {
          // shape: 'circle',
          indicator: [
            { name: '欧洲', max: 52000 },
            { name: '亚洲', max: 52000 },
            { name: '北美洲', max: 52000 },
            { name: '南美洲', max: 52000 },
            { name: '非洲', max: 52000 },
            { name: '澳洲', max: 52000 }
          ]
        },
        series: [
          {
            name: 'Budget vs spending',
            type: 'radar',
            data: [
              {
                value: [42000, 30000, 45800, 35000, 39000, 28000],
                name: '黑名单'
              },
              {
                value: [10000, 19000, 12000, 14000, 17000, 13000],
                name: '白名单'
              }
            ]
          }
        ]
      };

      option && myChart.setOption(option);
    },


  }

}
</script>

<style scoped>

</style>
