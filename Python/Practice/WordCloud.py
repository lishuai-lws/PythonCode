from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import cv2

if __name__ == '__main__':
    # os.system('export DISPLAY=:0.0')
    filename = "./txtfile/npuwiki.txt"
    # backgroundImage= plt.imread('./txtfile/npu.jpg')
    with open(filename,encoding='utf-8') as f:
        mytext = f.read()
        # print(mytext)
        wordcloud = WordCloud(width=800,height=400,background_color=None, mode="RGBA").generate(mytext)


    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.savefig('./txtfile/npuwiki.jpg')
    plt.show()
    print('词云已生成！')




# class wordcloud.WordCloud(font_path=None, width=400, height=200, margin=2,  

# ranks_only=None, prefer_horizontal=0.9,mask=None, scale=1, color_func=None,  

# max_words=200, min_font_size=4, stopwords=None, random_state=None,background_color='black',  

# max_font_size=None, font_step=1, mode='RGB', relative_scaling=0.5, regexp=None,  

# collocations=True,colormap=None, normalize_plurals=True)


# font_path : string //字体路径，需要展现什么字体就把该字体路径+后缀名写上，如：font_path = '黑体.ttf'
# width : int (default=400) //输出的画布宽度，默认为400像素
# height : int (default=200) //输出的画布高度，默认为200像素
# prefer_horizontal : float (default=0.90) //词语水平方向排版出现的频率，默认 0.9 （所以词语垂直方向排版出现频率为 0.1 ）
# mask : nd-array or None (default=None) //如果参数为空，则使用二维遮罩绘制词云。如果 mask 非空，设置的宽高值将被忽略，遮罩形状被 mask 取代。
# 除全白（#FFFFFF）的部分将不会绘制，其余部分会用于绘制词云。如：bg_pic = imread('读取一张图片.png')，
# 背景图片的画布一定要设置为白色（#FFFFFF），然后显示的形状为不是白色的其他颜色。可以用ps工具将自己要显示的形状复制到一个纯白色的画布上再保存，就ok了。
# scale : float (default=1) //按照比例进行放大画布，如设置为1.5，则长和宽都是原来画布的1.5倍。
# min_font_size : int (default=4) //显示的最小的字体大小
# font_step : int (default=1) //字体步长，如果步长大于1，会加快运算但是可能导致结果出现较大的误差。
# max_words : number (default=200) //要显示的词的最大个数
# stopwords : set of strings or None //设置需要屏蔽的词，如果为空，则使用内置的STOPWORDS
# background_color : color value (default=”black”) //背景颜色，如background_color='white',背景颜色为白色。
# max_font_size : int or None (default=None) //显示的最大的字体大小
# mode : string (default=”RGB”) //当参数为“RGBA”并且background_color不为空时，背景为透明。
# relative_scaling : float (default=.5) //词频和字体大小的关联性
# color_func : callable, default=None //生成新颜色的函数，如果为空，则使用 self.color_func
# regexp : string or None (optional) //使用正则表达式分隔输入的文本
# collocations : bool, default=True //是否包括两个词的搭配
# colormap : string or matplotlib colormap, default=”viridis” //给每个单词随机分配颜色，若指定color_func，则忽略该方法。
# fit_words(frequencies) //根据词频生成词云【frequencies，为字典类型】
# generate(text) //根据文本生成词云
# generate_from_frequencies(frequencies[, ...]) //根据词频生成词云
# generate_from_text(text) //根据文本生成词云
# process_text(text) //将长文本分词并去除屏蔽词（此处指英语，中文分词还是需要自己用别的库先行实现，使用上面的 fit_words(frequencies) ）
# recolor([random_state, color_func, colormap]) //对现有输出重新着色。重新上色会比重新生成整个词云快很多。
# to_array() //转化为 numpy array
# to_file(filename) //输出到文件
# 简要讲下几个会影响图像清晰问题的WordCloud的参数：
# mask：遮罩图，字的大小布局和颜色都会依据遮罩图生成。其实理论上这对字大小和清晰程度的影响不大，
# 但是遮罩图色和背景色background_color如果易混淆，则可能是一个导致看起来不清晰的因素；
# 另外遮罩图自身各个颜色之间的对比不强烈，也可能使图看起来层次感不够。
# 比如，一些图明度比较高，再加上背景白色，有可能导致字色太浅（背景色background_color又是白色）于是看起来不够“清晰”。
# background_color：背景色，默认黑。
# 这个本来其实也不怎么影响清晰度，但是，就像之前在mask中提到的，如果遮罩图像颜色过浅、背景设置白色，
# 可能导致字看起来“不清晰”。而实际上，我对一个浅色遮罩图分别用白、黑两种背景色后发现，
# 黑色背景的强烈对比之下会有若干很浅也很小的词浮现出来，而之前因背景色、字色过于相近而几乎无法用肉眼看出这些词。
# mode：默认“RGB”。根据说明文档，如果想设置透明底色的云词图，那么可以设置background_color=None, mode="RGBA"
# 但是！！！实际中我尝试设置透明背景色并没有成功过！
# 当我选取的遮罩图是白色底时，如果background_color设置为"white"或"black"时，生成的云词确实是对应的“白色”“黑色”；
# 但是按照上述参数设置透明色时，结果出来依然是白色。
# 当我选取的遮罩图是透明底时，那么不管我background_color设置为"white"或"black"，还是None加上mode="RGBA"，
# 结果都是把背景部分当做黑色图块，自动匹配黑色的字！——也就是并没有实现透明底的云词。
# 谁如果实现了透明底色的方案，欢迎给我留言。目前这个疑惑我打算先不研究了，放到以后再看。
# max_font_size：最大字号。源文件中也有讲到，图的生成会依据最大字号等因素去自动判断词的布局。
# 经测试，哪怕同一个图像，只要图本身尺寸不一样（比如我把一个300×300的图拉大到600×600再去当遮罩），那么同样的字号也是会有不同的效果。
# 原理想想也很自然，字号决定了字的尺寸，而图的尺寸变了以后，最大字相对于图的尺寸比例自然就变了。
# 所以，需要根据期望显示的效果，去调整最大字号参数值。

# min_font_size：最小字号。不设置的情况下，默认是4。
# 尝试了设置比4大的字号，例如8、10，结果就是原本小于设定值且大于4号的词都直接不显示了，其它内容和未设置该值时都一样。

# relative_scaling：表示词频和云词图中字大小的关系参数，默认0.5。
# 为0时，表示只考虑词排序，而不考虑词频数；为1时，表示两倍词频的词也会用两倍字号显示。

# scale：根据说明文档，当云词图很大的，加大该值会比使用更大的图更快，但值越高也会越慢（计算更复杂）。
# 默认值是1。实际测试中，更大的值，确实输出图像看起来更精细（较小较浅的词会颜色更重，也感觉清楚，大的词差异不明显）。
# 不过，可能由于我选的图不大、词也没有很多，所以差距并没有很大，缩小排列一下就基本上辨别不出多少差别了。
