<template>
  <div>
    <el-carousel :interval="4000" type="card" height="350px" >
      <el-carousel-item v-for="(item, index) in list" :key="index" style="border-radius: 15px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);">
        <!-- 使用v-bind绑定图片src属性 -->
        <img :src="item.src" alt="轮播图片" class="carousel-image">
      </el-carousel-item>
    </el-carousel>
  </div>

</template>

<script>
export default {
  name: "List",
  data(){
    return {
      list:[
        { src: require('@/assets/image/1.jpg') },
        { src: require('@/assets/image/2.jpg') },
        { src: require('@/assets/image/3.jpg') },
        { src: require('@/assets/image/4.jpg') },
        { src: require('@/assets/image/5.jpg') },
        { src: require('@/assets/image/6.jpg') },
      ],
    };
  }


}
</script>

<style>
.el-carousel__item h3 {
  color: #475669;
  font-size: 14px;
  opacity: 0.75;
  line-height: 200px;
  margin: 0;
  border-radius: 10px;
}

.el-carousel__item:nth-child(2n) {
  background-color: #99a9bf;
}

.el-carousel__item:nth-child(2n + 1) {
  background-color: #d3dce6;
}

/* 新增的图片样式 */
.carousel-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  /* 添加圆角效果 */
  border-radius: 15px; /* 这里设置的圆角半径为10px，你可以根据需要调整数值 */

  /* 添加阴影效果 */

}
</style>
