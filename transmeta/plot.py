import matplotlib.pyplot as plt

# 数据
x = [5, 10, 25]
y = [1.6, 2.2, 3.5]

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o-', linewidth=2, markersize=8)
plt.xlabel('横坐标 (%)')
plt.ylabel('纵坐标')
plt.title('三个数据点的折线图')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()