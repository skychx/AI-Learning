from layer_naive import Mulayer


apple = 100
apple_num = 2
tax = 1.1

# layer
# mul_apple_layer -> mul_tax_layer
mul_apple_layer = Mulayer()
mul_tax_layer = Mulayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print("price:", int(price)) # 220

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("dTax:", dtax) #200
print("dApple_num:", int(dapple_num)) # 110
print("dApple:", dapple) # 2.2