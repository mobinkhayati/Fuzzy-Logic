import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# تعریف محدوده ورودی
x = np.arange(-5, 5, 0.1)

# تعریف پارامترهای تابع عضویت گوسی
mean = 0.0  # میانگین
sigma = 1.0  # انحراف معیار

# تعریف تابع عضویت گوسی
membership = fuzz.gaussmf(x, mean, sigma)

# نمایش تابع عضویت گوسی
plt.figure()
plt.plot(x, membership, 'b', linewidth=1.5)
plt.title('Gaussian Membership Function')
plt.ylabel('Membership')
plt.xlabel('input variable (x)')
plt.show()
