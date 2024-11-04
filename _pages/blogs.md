---
title:  "Blogs"
layout: archive
permalink: /Blogs/
header:
  overlay_image: /assets/images/about/night.jpg
author_profile: true
comments: true
---

These blogs are written to log my study during research for future reference. Hope you can also enjoy! 

<ul>
  {% for post in site.posts %}
    {% unless post.next %}
      <font color="#778899"><h2>{{ post.date | date: '%Y %b' }}</h2></font>
    {% else %}
      {% capture year %}{{ post.date | date: '%Y %b' }}{% endcapture %}
      {% capture nyear %}{{ post.next.date | date: '%Y %b' }}{% endcapture %}
      {% if year != nyear %}
        <font color="#778899"><h2>{{ post.date | date: '%Y %b' }}</h2></font>
      {% endif %}

    {% endunless %}
   {% include archive-single.html %}
  {% endfor %}
</ul>

