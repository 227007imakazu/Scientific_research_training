import Vue from 'vue'
import VueRouter from 'vue-router'
import HomeView from '../views/HomeView.vue'
import IpSearch from "@/views/IpSearch.vue";
import Index from "@/views/Index.vue";
import List from "@/views/List.vue";
import DataDetection from "@/views/DataDetection.vue";

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
    children: [
      {
        path: '/IpSearch', //查询后的页面
        name: 'ipSearch',
        component: IpSearch
      },
      {
        path: '/Index', //首页数据可视化
        name: 'index',
        component: Index
      },
      {
        path: 'DataDetection', //数据包分析
        name: 'dataDetection',
        component: DataDetection
      },
      {
        path: 'List', //黑白名单查询
        name: 'list',
        component: List
      },
    ]
  },
]

const router = new VueRouter({
  routes
})

export default router
