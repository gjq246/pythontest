#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import support

# 现在可以调用模块里包含的函数了
support.print_func("Runoob");

# 定义函数
def printme( str ):
   "打印任何传入的字符串"
   print str;
   return;
 
# 调用函数
printme("我要调用用户自定义函数!");
printme("再次调用同一函数");