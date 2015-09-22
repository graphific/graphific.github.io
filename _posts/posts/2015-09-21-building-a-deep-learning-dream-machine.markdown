---
layout: post
title: "Building a Deep Learning (Dream) Machine"
modified:
categories: posts
excerpt: Some pointers on the slippery path towards building your own machine for Deep Learning
tags: [diy, deep learning, hardware, graphic cards]
comments: true
image:
  feature:
date: 2015-09-21T15:19:36+02:00
---

![DL System]({{ site.url }}/images/dlpc/machine_sm.jpg)
{: .image-right}
> *As a* <a href="http://www.csc.kth.se/~roelof/">PhD student in Deep Learning</a>, *as well as running my own* <a href="http://www.graph-technologies.com/">consultancy, building machine learning products for clients</a> *I'm used to working in the cloud and will keep doing so for production-oriented systems/algorithms. There are however huge drawbacks to cloud-based systems for more research oriented tasks where you mainly want to try out various algorithms and architectures, to iterate and move fast. To make this possible I decided to custom design and build my own system specifically tailored for Deep Learning, stacked full with GPUs. This turned out both more easy and more difficult than I imagined. In what follows I will share my "adventure" with you. I hope it will be useful for both novel and established Deep Learning practitioners.*

<br>

If you're like me, and working day (and night) with practical machine learning applications, you know the pain of not having the right hardware for the task at hand. Whether you're working in industry or academia, nothing is more annoying than having to wait longer than necessary for the results of an experiment or calculation to come in. Fast hardware is a must for productive research and development, and GPUs are often the main bottleneck, especially so for Deep Neural Nets (DNNs).


Yes, it's true: Cloud providers like Amazon offer GPU capable instances for <a href="https://aws.amazon.com/ec2/pricing/">under $1/h</a> and production-ready <a href="http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html">virtual machines</a> can be exported, shared, and reused. If installing libraries from scratch is more your thing, you probably know that both software and hardware libraries can easily be installed with regularly updated <a rel="dlcloudscripts" href="https://github.com/deeplearningparis/dl-machine">install scripts</a>  or <a href="https://hub.docker.com/r/tleyden5iwx/ubuntu-cuda/">dockerized containers</a>. So far so good. What however about the kinds of applications which need more than the 4GB GPUs Amazon offers (even their newest <a href="https://aws.amazon.com/blogs/aws/new-g2-instance-type-with-4x-more-gpu-power/">g2.8xlarge</a> still offers the same 4GB GPUs, be it x4)? The few other cloud providers offering bigger GPU's (6GB generally) all seem to be either too custom tailored for very specific applications (video edition or biosci), or just completely unusable.

<br>
So what is one to do? Simple: <b>get</b> your own GPU rig!


{% include _toc.html %}

## Know your stuff: Research
Once I decided it was time to get my own GPU system I first thought: why go through the hassle of building one yourself, hasn't Nvidia just released its glorious <a href="https://developer.nvidia.com/devbox">DevBox</a>, and might there not be other vendors doing the same for Deep Learning applications? Well yes it turns out there are <a href="https://www.microway.com/preconfiguredsystems/whisperstation-deep-learning/">some</a> other companies building research-oriented machines, but none of them ships or sells to Europe. Nvidia's Devbox also only ships to the USA, next to being ridiculously overpriced (with its $15k for around <a href="http://pcpartpicker.com/p/NP4MNG">$9k of hardware components</a>), as well as has a huge waiting list.

<br>
Again, what is one to do? Simple: <b>build</b> you're own GPU rig!


## Starting out: Choosing the right components

Surfing the web, I found <a href="https://timdettmers.wordpress.com">Tim Dettmers' blog</a> where he has a couple of <a href="https://timdettmers.wordpress.com/2014/08/14/which-gpu-for-deep-learning/">hugely</a> <a href="https://timdettmers.wordpress.com/2015/03/09/deep-learning-hardware-guide/">useful</a> <a href="https://timdettmers.wordpress.com/2014/09/21/how-to-build-and-use-a-multi-gpu-system-for-deep-learning/">posts</a> on which GPU's and hardware to choose for Deep Learning applications. I won't repeat this information in full here. Just go and check them out! Both the posts and the comments are very much worth to take a look at.

In short:

* Double precision (as Nvidia's <a href="http://www.nvidia.com/object/tesla-workstations.html">Tesla K20/40/80</a> offer) is a waste of money as this type of precision is not needed for DNNs;
* Think of how many <b>GPUs</b> you might want to have, now and in the future. Four GPUs is the max as anything more will not really give much performance benefits anymore. This is mainly because the best motherboards can only support a maximum of 40 lanes (with a 16x8x8x8 configuration). Furthermore, every GPU adds a certain amount of overhead, where your system has to decide which GPU to use for which task;
* Get a <b>motherboard</b> which supports PCIe 3.0 and supports PCIe power connectors of 8pin + 6pin with one cable, so you can add up to 4 GPUs. The motherboard should be able to support your GPU configuration, ie enough physicial lanes to support a x8/x8/x8/x8 setup for 4 GPUs;
* Get a chassis with enough space for everything. Bigger chassis offer more airflow.
Make sure there are enough <b>PCIe slots</b> to support all the GPUs, as well as possibly any other PCIE cards you might install (as fast Gigabit network cards or whatever). One GPUs typically takes the space of 2 PCIe slots. In a typical chassis this means 7 PCIe slots, as the last GPU can be mounted at the bottom using only one slot;
* <b>CPU's</b> don't have to be super fast and don't have to have a massive amount of cores. Just make sure you get at least as many cores as you might have GPUs, again now and in the future (Intel CPUs generally have 2 threads for each 1 core). Also make sure the CPU supports 40 PCIe lanes, some new Haswell CPUs only support 32;
* Get twice the amount of <b>RAM</b> as your total GPU memory;
* <b>SSD</b> is nice but only an absolute necessity if you load datasets which don't fit into GPU memory and RAM combined. If you do get an SSD, get at least one larger than your largest dataset;
* As for ordinary <b>mechanical hard disks</b>, you might want to get plenty of disk space to store all your datasets and other types of data. <a href="https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_5">RAID5</a> is nice if you have at least 3 disks of the same size. Basically upon failure of a single drive you won't lose your data. Other RAID configurations like RAID0 for performance boost usually not of much use: You have SSDs for speed, and these are already faster than your GPU can load data from them through its PCIE bandwidth;
* As for the <b>Power Supply Unit (PSU)</b> get one with as high efficiency as you can afford to, and take into account the total wattage you might need - again - now and in the future: Titanium or platinum quality PSUs are worth the money: you will save money and the environment, and get back the extra $$ in no time on saved energy costs. 1500 to 1600 Watt is what you probably need for a 4 GPU system;
* <b>Cooling</b> is super important, as it affects both performance and noise. You want to keep the temperature of a GPU at all times below 80 degrees. Anything higher will make the unit lower its voltage and take a hit in performance. Furthermore, too hot temperatures will wear out your GPU; Something you'd probably want to avoid. As for cooling there are two main options: Air cooling (Fans), or Water cooling (pipes):
    * <b>Air cooling</b> is cheaper, simple to install and maintain, but does make a hell lot of noise;
    * <b>Water cooling</b> is more expensive, tough to install correctly, but does not make any noise, and cools the components attached to the water cooling system much much better. You would want some chassis fans anyway to keep all the other parts cool, so you'd still have some noise, but less than with a fully air cooled system.


## Putting it all together

In the end, after thorough reading, <a href="https://timdettmers.wordpress.com/2015/03/09/deep-learning-hardware-guide/comment-page-1/#comment-540">helpful replies</a> from Tim Dettmers, and also going over Nvidia's DevBox and Gamer Forums, the components I chose to put together. It is clear that the machine is partly (at least the chassis is) inspired by Nvidia’s DevBox, but for almost 1/2 of the price.

* Chassis: <a href="http://www.corsair.com/en/carbide-series-air-540-high-airflow-atx-cube-case">Carbide Air 540 High Airflow ATX Cube</a>
* Motherboard: <a href="https://www.asus.com/us/Commercial_Servers_Workstations/X99E_WS/">Asus X99-E WS workstation class motherboard</a> with <b>4-way PCI-E Gen3 x16 support</b>
* RAM: 64GB DDR4 Kingston 2133Mhz (8x8GB)
* CPU: Intel(Haswell-e) Core <a href="http://ark.intel.com/products/82931/Intel-Core-i7-5930K-Processor-15M-Cache-up-to-3_70-GHz">i7 5930K</a> (6 Core 3.5GHz)
* GPUs: 3 x NVIDIA GTX TITAN-X 12GB
* HDD: 3 X 3TB WD Red in RAID5 configuration
* SSD: 2 X 500GB SSD Samsung EVO 850
* PSU: <a href="http://www.corsair.com/en/ax1500i-digital-atx-power-supply-1500-watt-fully-modular-psu">Corsair AX1500i</a> (1500Watt) 80 Plus Titanium (94% energy efficiency)
* Cooling: Custom (soft piped) Water Cooling for both the CPU and GPUs: a refilling hole drilled in the top of the chassis, and transparent reservoir in the front (see pictures below)

<figure class="third">
    <a href="/images/dlpc/1_building.jpg"><img src="/images/dlpc/1_building.jpg"></a>
    <a href="/images/dlpc/2_pc.jpg"><img src="/images/dlpc/2_pc.jpg"></a>
    <a href="/images/dlpc/3_pc_side.jpg"><img src="/images/dlpc/3_pc_side.jpg"></a>
    <figcaption>a beautiful sight... left: The system is being built. You can see the plastic piping for the water cooling going through the holes already available in the Carbide Air 540 chassis. The motherboard is vertically mounted.<br/> middle & right: The system is completely built. Notice that the water reservoir can be seen from the outside. Red plastic pipes can be seen going from up (there is a filling hole on the outside), down to the <a href="http://www.highflow.nl/watercooling-sets/cpu-sets/xspc-raystorm-d5-rx360-watercooling-kit.html">water pump</a>, through the <a href="http://www.highflow.nl/water-blocks/gpu-blocks/nvidia/ek-fc-titan-x.html">water blocks</a> installed on the GPUs (keeping these cool). A similar thing happens for the CPU which has its separate cool block and pipes leading to and from it.</figcaption>
</figure>

## Building it yourself (DIY) or asking for help

### Option A: DIY

If you have the time and willpower to build an entire system yourself, of course, this is the best way to fully understand how components work and which types of hardware fit well together. Also, you might better know what to do when a component fails, and can replace or repair it more easily.

### Option B: Outside help

Another option is asking a specialized company to order the parts and build the entire system for you. The kinds of companies you want to be looking for is Gamer PC companies, which are used to custom build systems for gamers. They might even have experience with water cooled systems, although for gamer PCs one usually only water cools the CPU, and there are handy premade kits for that. This is, of course, different for full on water cooled systems where also multiple GPUs need to be screwed open, heatsinks placed on top, and the water piping, compressor caps, bits, and whatnot need to be all properly put together. The worst thing after all your hard work would be to have a water leak in your system, and damage to your GPU or other components.

Mainly because I couldn't see myself properly put together all the necessary components for water cooling, as well as lack of time to read up on the full procedure, I opted for the second option, and found a very capable hardware builder to help me out with putting the first version of my Deep Learning Machine together.
If you don't mind having your PC built in the Netherlands, I can fully recommend <a href="http://www.computer-bestel.nl/">computer-bestel.nl</a>. You can see what they usually have in stock for high-end systems <a href="http://www.computer-bestel.nl/game-computer-samenstellen/ultimate-intel-game-computer/">here</a>, but you probably want to give a call or mail Johan Oosterhuis, computer-bestel's founder.
If you're as crazy like me to go for a water cooled system take in mind that it's rather fragile and therefore not recommended to ship it by ordinary parcel mail. A system like the one I have (let) build also will be too big to be counted as an "instrument" by airlines, so you can't take it with you in the plane either, so transport might be something to think about before actually building your system.

<br>
*Thats it for now. Next time (next week or so?) we'll cover a lot of ground, installing CUDA support, and everything you need on a software level to get your bare metal system running some Deep Neural Nets!*
