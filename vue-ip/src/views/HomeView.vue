<template>
  <div id="app">
    <el-container>
      <!--      头部-->
      <el-header class="el-header">

        <div class="left">
          <img src="@/assets/logo.png" class="logo" alt="" style="width: 64px;height: 64px">
          <img src="@/assets/Title.png" class="logo" alt="">
        </div>
        <div class="middle">
          <el-input
              placeholder="请输入要查询的IP地址" v-model="inputSearch" class="inputSearch" style="width: 500px; height: 40px;font-size: 18px">
            <i slot="prefix"></i>
          </el-input>
          <el-button type="primary" icon="el-icon-search" class="searchButton" style="font-size: 18px">搜索</el-button>
        </div>
      </el-header>
    </el-container>
    <!--      侧边栏-->
    <el-container  class="el-container" style="height: 100%">
      <el-aside class="el-aside" width="250px">
        <el-scrollbar >
          <el-menu :default-openeds="[]" router>
            <el-menu-item index="Index" class="menu" @click="setActive('Index')">
              <i class="el-icon-help"></i> 首页
            </el-menu-item>
            <el-menu-item index="DataDetection" class="menu" @click="setActive('DataDetection')">
              <i class="el-icon-upload"></i> 数据检测
            </el-menu-item>
            <el-menu-item index="List" class="menu" @click="setActive('List')">
              <i class="el-icon-s-custom"></i> 黑白名单查询
            </el-menu-item>
          </el-menu>
        </el-scrollbar>
      </el-aside>
      <el-main class="el-main" v-loading.fullscreen.lock="fullscreenLoading">
        <router-view></router-view>
      </el-main>
    </el-container>
  </div>
</template>

<script>
export default {
    data() {
      return {
        inputSearch: '',
        activeIndex: '',
        fullscreenLoading: false,
      }
    },
  methods: {
    setActive(index) {
      this.activeIndex = index;
    },
    // 新增一个方法用于跳转到Index页面
    goToIndex() {
      this.$router.push('/Index');
    },
    openFullScreen1() {
      this.fullscreenLoading = true;
      setTimeout(() => {
        this.fullscreenLoading = false;
      }, 500);
    }
  },
  mounted() {
      this.goToIndex();
      this.openFullScreen1();
  }
}

</script>
<style scoped>
.logo{
  width: 350px;
  height: 64px;
  object-fit: cover;
  margin-left: 20px;
  position: relative;

}
.name{
  font-size: 25px;
  margin-left: 15px;
  margin-top: 15px;
  color: beige;
  font-family: Arial, Helvetica, sans-serif;
}
.el-header{
  background-color: #909399;
  color: #333;
  height: 80px!important;
  padding: 0;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.el-aside {
  background-color: #909399;
  color: #333;
  text-align: left;
  line-height: 200px;
  width: 250px; /* 设置固定宽度 */
  flex: 0 0 250px; /* 不允许伸缩，保持固定宽度 */
  height: 100vh; /* 设置高度为视口高度 */

}
.el-main {
  background-color: #E9EEF3;
  color: #333;
  text-align: center;
  line-height: 160px;
  flex: 1; /* 占据剩余空间 */
  height: 100vh; /* 设置高度为视口高度 */
  padding: 0;
  margin: 0;
}
.left{
  position: relative;
  display: flex;
  left: 0;
}
.middle{
  margin: 0 auto;
  height: 40px;
}
.searchButton{
  margin-left: 20px;
}
.menu{
  font-size: 20px;
  color: cadetblue;
  justify-items: center;
  border-radius: 4px;
  transition: all 0.3s ease;
  /*box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);*/
}
.menu[data-v-active-index="Index"] {
  color: white; /* 当Index菜单项被选中时的颜色 */
}
.menu[data-v-active-index="DataDetection"] {
  color: white; /* 当DataDetection菜单项被选中时的颜色 */
}
.menu[data-v-active-index="List"] {
  color: white; /* 当List菜单项被选中时的颜色 */
}


</style>
