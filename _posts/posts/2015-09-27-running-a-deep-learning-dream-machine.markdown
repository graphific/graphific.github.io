---
layout: post
title: "II: Running a Deep Learning (Dream) Machine"
modified:
categories: posts
excerpt: Building a machine for Deep Learning is one thing, getting it to run the right software another...
tags: [diy, deep learning, software]
comments: true
image:
  feature:
date: 2015-09-27T22:10:25+02:00
---

![DL System]({{ site.url }}/images/dlpc/machine_sm.jpg)
{: .image-right}
> *As a* <a href="http://www.csc.kth.se/~roelof/">PhD student in Deep Learning</a>, *as well as running my own* <a href="http://www.graph-technologies.com/">consultancy, building machine learning products for clients</a> *I'm used to working in the cloud and will keep doing so for production-oriented systems/algorithms. There are however huge drawbacks to cloud-based systems for more research oriented tasks where you mainly want to try out various algorithms and architectures, to iterate and move fast. To make this possible I decided to custom design and build my own system specifically tailored for Deep Learning, stacked full with GPUs. This turned out both more easy and more difficult than I imagined. In what follows I will share my "adventure" with you. I hope it will be useful for both novel and established Deep Learning practitioners.*

See the earlier part of this series at: 

* <a href="/posts/building-a-deep-learning-dream-machine/">part 1: Hardware</a>
<br>

{% include _toc.html %}

## Software & Libraries

Now we got a bare metal system, it's time to install some software! There are a couple of really good posts on installing all the tools and libraries for Deep Learning. To make things easy, I've put some gists together for the occasion. It will help you to install the Nvidia CUDA drivers, as well as the kind of libraries and tools I tend to work with for Deep Learning. It is assumed that you have installed Ubuntu 14.04.3 LTS as your operating system.


### 1 Installing CUDA

Getting the graphical drivers to work can be a pain in the %€#. My issue at the time was that Titan X GPUs are only supported from Nvidia 346 onward, but that these drivers wouldn't work with my specific monitor. With some xconfig modding I got it somehow to work on higher resolutions than 800x600, using <a href="http://www.nvidia.com/download/driverResults.aspx/87650/en-us">352.30</a> as the graphical driver.

The script installs CUDA 7.0. I chose to install the newest CUDA 7.5. While this version does offer some improvements, it also tends to be a bit difficult to get working for some libraries. If you want to be fast up and running, try 7.0 instead.
{: .notice}

{% gist graphific/e74d33f837d742a17334 %}

### 2 Testing CUDA

Done? Great, let's see if the CUDA drivers work. Go to the CUDA sample directory, _make_ them and run _./deviceQuery_. Your GPUs should be shown.

![Querying GPUs]({{ site.url }}/images/dlpc/devicequery.png)

{% gist graphific/5df4a3a8ba580cabf8e4 %}

### 3 Deep Learning Libraries

Ok final step: we're getting to the fun part: Choosing which Deep Learning libraries to work with is a matter of personal preference, as well as domain given.

* <a href="http://deeplearning.net/software/theano/">Theano</a> gives you the most freedom as a researcher to do whatever you want to do. You get to implement many things yourself, and in so doing gain a deep understanding of how DNNs work, but maybe not the best suited for a beginner who wants to first "play" a bit.
* Personally I'm a big fan of both <a href="http://keras.io/">Keras</a> (main contributor: <a href="https://twitter.com/fchollet">François Chollet</a>, who just moved to work at Google) and <a href="http://lasagne.readthedocs.org/en/latest/">Lasagne</a> (team of eight people, but main contributor: <a href="https://twitter.com/sedielem">Sander Dielemans</a>, who recently finished his PhD and now works at Google Deepmind). They have a good level of abstraction, are actively developed, but also offer easy ways to plug in your own modules or code projects.
* <a href="http://torch.ch/">Torch</a> can be challenging if you're used to Python as you'd have to learn Lua. After working for some time with Torch, I can say its actually a pretty good language to work with. Its only real big issue is that it's tough to interface from other languages to Lua. Also for research purposes Torch will do fine, but for production level pipelines, Torch is hard to test and seems to completely lack any type of error handling. On the positive side though: It has support for CUDA, and has many packages to play with. Torch also seems to be the most widely adopted library in the industry. Facebook (<a href="http://ronan.collobert.com/">Ronan Collobert</a> & <a href="https://twitter.com/amiconfusediam">Soumith Chintala</a>), Deepmind (<a href="http://koray.kavukcuoglu.org/">Koray Kavukçuoğlu</a>) and also Twitter (<a href="https://twitter.com/clmt">Clement Farabet</a>) are all active contributors.
* <a href="http://caffe.berkeleyvision.org/">Caffe</a> (originally by <a href="http://daggerfs.com/">Yangqing Jia</a> as his PhD work), the former dominant Deep Learning framework (mainly used for Convnets) from Berkeley is still widely used and a nice framework to start with. The separation between training regimes (solver.prototxt), and architecture (train_val.prototxt) files allows for easy experimentation. I have found Caffe also to be the only framework which supports multi-GPU use out of the box. Without much hassle, one can just pass the _--gpu all_ or _--gpu &lt;id&gt;_ argument to use all available GPUs.
* <a href="http://blocks.readthedocs.org/en/latest/">Blocks</a> is a bit more recent python-based framework. It has pretty nice separation of modules you can write yourself, and which are called "Bricks". Especially its partner "Fuel" is a really nice way to deal with data: Fuel is a wrapper for many existing or your own datasets. It implements "iteration schemes" to stream your data to your model and "transformers" for all kinds of typical data transformations and pre-processing steps.
* <a href="https://github.com/NervanaSystems/neon">neon</a> is Nervana Systems' Python based Deep Learning framework, build on top of <a href="https://github.com/NervanaSystems/nervanagpu">Nervana's gpu kernel</a> (an alternative to Nvidia's CuDNN). It is the only framework running this specific kernel, and <a href="https://github.com/soumith/convnet-benchmarks">latest benchmarks</a> show it to be the fastest for some specific tasks.

<figure class="half">
    <a href="/images/dlpc/python-for-image-understanding-deep-learning-with-convolutional-neural-nets-30-1024.jpg"><img src="/images/dlpc/python-for-image-understanding-deep-learning-with-convolutional-neural-nets-30-1024.jpg"></a>
    <a href="/images/dlpc/python-for-image-understanding-deep-learning-with-convolutional-neural-nets-31-1024.jpg"><img src="/images/dlpc/python-for-image-understanding-deep-learning-with-convolutional-neural-nets-31-1024.jpg"></a></a>
    <figcaption>Another way to represent some of the (python-oriented) Deep Learning libraries: from lower level DIY, to a higher level, more functional. (From a <a href="http://www.slideshare.net/roelofp/python-for-image-understanding-deep-learning-with-convolutional-neural-nets">talk given at PyData Conf, London 2015</a></figcaption>
</figure>
<br>

Ready? The script below will install Theano, Torch, Caffe, Digits, Lasagne and Keras. We haven't managed <a href="https://developer.nvidia.com/digits">Digits</a> before, but its a graphical web interface build on top of Caffe. It's pretty basic, but if you're just starting out it's an easy way to train some ConvNets and build some image classifiers. 

{% gist graphific/f211174ebffb1f874f6d %}


## Where to go from here? Resources and Links

If you managed to come this far, congratulations! It's time to play! There are many tutorials, seminars, articles, and web pages out there to get you started with Deep Learning. Some links below to get you on your way:

* The single best resource is <a href="http://deeplearning.net/">deeplearning.net</a>, especially the <a href="http://deeplearning.net/tutorial/">tutorial</a> section. (On a side note: if you're really into Deep Learning and want to do a PhD, there's a great <a href="http://deeplearning.net/deep-learning-research-groups-and-labs/">deep learning research groups and labs</a> as well);
* There are multiple Deep Learning Meetup groups around the globe with interesting talks, and the kind of people you might want to interact with: <a href="www.meetup.com/Stockholm-Deep-Learning-Meetup/">Stockholm</a>, <a href="http://www.meetup.com/deeplearning/">Munich</a>, <a href="http://www.meetup.com/Deep-Learning-London/">London</a>, <a href="http://www.meetup.com/Deep-Learning-Paris-Meetup/">Paris</a>, <a href="http://www.meetup.com/SF-Neural-Network-Afficianados-Discussion-Group/">San Francisco</a> and <a href="http://www.meetup.com/Deep-Learning-Toronto-Meetup/">Toronto</a>;
* Videos from the <a href="http://videolectures.net/deeplearning2015_montreal/">Deep Learning Summer School in Montreal 2015</a>;
* Nando de Freitas <a href="https://www.youtube.com/watch?v=PlhFWT7vAEw&list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu&index=16">series of Deep Learning video lectures</a> at Oxford;
* CS224d: Deep Learning for Natural Language Processing at Stanford, <a href="http://cs224d.stanford.edu/syllabus.html">syllabus</a> with lecture slides and links to all video recordings;
* CS231n: Convolutional Neural Networks for Visual Recognition at Stanford by Fei-Fei Li and Andrej Karpathy, <a href="http://cs231n.github.io/">course notes</a> with lecture slides and assignments;
* A lot of Deep Learning news gets send around over the web through the <a href="https://plus.google.com/u/0/communities/112866381580457264725">Deep Learning Google+ community</a>, as well as on Twitter under the hashtag <a href="https://twitter.com/search?q=%23dlearn&src=typd">#dlearn</a>;
* The biggest Deep Learning <a href="http://memkite.com/deep-learning-bibliography/">Bibliography at Memkite</a>;
* And further of course just keep checking <a href="http://arxiv.org/">arxiv</a>, or a more recent project linking articles to code: <a href="http://gitxiv.com/">GitXiv</a>;
* Most deep learning libraries like Caffe and others have their own mailing lists or at least GitHub issues pages where discussions take place.

<br>
*That's it for now. Next time we will benchmark different numbers of GPUs and nets, and learn more about the kinds of speedups multiple GPUs can offer.*
