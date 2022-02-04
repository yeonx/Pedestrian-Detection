# Pedestrian Detection

*참고 자료*  
[SSD 논문](https://arxiv.org/abs/1512.02325)  
[SSD GitHub](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)  
[half-way 논문](http://www.kibme.org/resources/journal/20180731114000412.pdf)

[나의 SSD 논문 코드 정리](https://github.com/yeonx/Pedestrian-Detection/blob/main/URP%20SSD%20%EC%9D%B4%EB%A1%A0_%EC%BD%94%EB%93%9C.pdf)

 **문제발견**
<ol type = '1'>
 

  <li> Thermal와 RGB에 따른 성능차이 -> day와 night의 성능 차이 </li><ul>
    <li>Thermal -> MR(all): 29.20 MR(day): 35.76 MR(night): 15.79</li>
  <li>RGB -> MR(all): 35.03 MR(day): 32.57 MR(night): 41.75</li></ul>
  
  <li> 나무나 차의 문, 그 외의 것들을 사람으로 인식하는 경우들 </li>
  <img src="https://user-images.githubusercontent.com/71878202/128730420-89f6433f-d73a-4a7a-ba13-fd3faab5ce85.png"  width="800" height="300">
  <li> 물체에 가려진 사람을 인식하지 못하는 경우 </li>
  <img src="https://user-images.githubusercontent.com/71878202/128727256-d7353d9f-8b84-417b-997c-5e8b806e7c92.jpg"  width="400" height="300">
 
  <li> 멀리 있는 사람을 인식하지 못하는 경우 x </li>
  
 </ol>

-> _**1.2에 초점을 맞추어 feature fusion을 이용하여 성능 개선을 시도**_
![image](https://user-images.githubusercontent.com/71878202/128730631-92d62e8d-c0bf-4094-a493-bca4f45acdec.png)

---
**Halfway fusion**

![image](https://user-images.githubusercontent.com/71878202/128731126-429b7c2b-30c1-4f76-9c48-fb9f481b46c9.png)

<ol type = '1'>
 <li> day와 night의 성능 차이 개선 </li> <ul>
 <li> earlyway -> MR(all): 23.03 MR(day): 24.42 MR(night): 20.57 </li>
 <li>halfway -> MR(all): 19.86 MR(day): 20.96 MR(night): 17.30</li> </ul>
 <li> 나무나 차의 문, 그 외의 것들을 사람으로 인식하는 경우들 개선 </li>
 <img src="https://user-images.githubusercontent.com/71878202/128731924-cf8298a0-2fb0-4f66-9850-ffe5e4b4c0e6.png"  width="800" height="300">
 <img src="https://user-images.githubusercontent.com/71878202/128732207-9edd04bb-d527-4089-b71e-614519596c0a.png"  width="800" height="300">
 <img src="https://user-images.githubusercontent.com/71878202/128732320-b4feb640-c3b3-4cb2-b29f-046febbdf0e8.png"  width="800" height="300">
 </ol>
 
 ---
