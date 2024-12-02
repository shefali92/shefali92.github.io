---
title: Let's Paint a Little :) 
description: Some of my Art Works
date: 2024-01-01
layout: "simple"
---
<style>
  /* Global Styles */
  body {
    font-family: 'Arial', sans-serif; /* Clean sans-serif font */
    background-color: #f4f4f4; /* Light gray background */
    color: #333; /* Dark text for readability */
    margin: 0;
    padding: 0;
    line-height: 1.6;
    text-align: center; /* Center all text */
  }

  /* Heading Style */
  h1 {
    font-family: 'Dancing Script', cursive; /* Fun yet classy cursive font */
    font-size: 2rem;
    color: #222;
    /* margin-top: 12px;
    margin-bottom: 0px; */ 
    /* letter-spacing: 2px; */
    /* text-transform: uppercase; */
    /* font-weight: bold; */
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); /* Soft shadow for elegance */
  }

  h2 {
    font-family: 'Dancing Script', cursive; /* Fun yet classy cursive font */
    font-size: 1rem;
    color: #555;
    /* margin-top: 10px; */
    /* letter-spacing: 2px; */
    /* text-transform: uppercase; */
    /* font-weight: bold; */
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); /* Soft shadow for elegance */
  }

  /* Art Grid Container */
  .art-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr); /* Default 3 columns */
    gap: 40px;
    max-width: 1200px;
    margin: 60px auto;
    padding: 0 20px;
  }

  /* Individual Art Items */
  .art-item {
    background-color: #fff;
    border: 1px solid #ddd; /* Light border for each item */
    border-radius: 8px;
    padding: 20px;
    transition: transform 0.3s, box-shadow 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .art-item:hover {
    transform: translateY(-8px); /* Slight lift effect */
    box-shadow: 0 12px 20px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
  }

  /* Image Styling */
  .art-item img {
    width: 100%;
    height: auto;
    border-radius: 6px;
    object-fit: cover; /* Ensures the image covers the space neatly */
    transition: transform 0.3s ease;
    margin-bottom: 20px;
  }

  .art-item img:hover {
    transform: scale(1.05); /* Slight zoom-in effect on hover */
  }

  /* Landscape images (abstract.jpg, Dream.jpg, friends.jpg) */
  .landscape {
    grid-column: span 2; /* Make landscape images span across 2 columns */
    width: 100%;
    height: auto;
  }

  /* Special class for smaller Dream.jpg */
  .smaller-dream {
    grid-column: span 1; /* Dream.jpg will occupy only 1 column */
    width: 100%; /* Ensure it stays within its column */
    height: auto;
  }

  /* Adjust the height of landscape images */
  .art-item.landscape img {
    max-height: 350px; /* Set max height for landscape images to make them shorter */
  }

  /* Caption Styling */
  .art-item p {
    font-size: 1rem;
    color: #333;
    margin-top: 8px;
    /* text-transform: uppercase; */
    font-weight: 400;
    /* letter-spacing: 1px; */
    font-style: italic;
  }

  /* Hover effect for title text */
  .art-item:hover p {
    color: #ff6347; /* Change text color on hover */
  }

  /* Responsive Design for smaller screens */
  @media (max-width: 768px) {
    .art-grid {
      grid-template-columns: 1fr; /* Stack the images in a single column on smaller screens */
      padding: 10px;
    }
    .landscape {
      grid-column: span 1; /* In smaller screens, landscape images take up 1 column */
    }
  }
</style>
<!-- 
<h2>Some of my Art Works over the years :)</h2> -->

<div class="art-grid">
  <div class="art-item">
    <img src="../art/inevitable.jpg" alt="The Inevitable" />
    <p>Title: The Answer To Everything :)<br/>Medium: Acrylic</p>
  </div>
  <div class="art-item">
    <img src="../art/crown.jpg" alt="The Crown" />
    <p>Title: Wear Your Crown<br/>Medium: Acrylic</p>
  </div>
  <div class="art-item">
    <img src="../art/ganesha.jpg" alt="Ganesha" />
    <p>Title: The Beginnings With Lord Ganesha<br/>Medium: Acrylic</p>
  </div>
  <div class="art-item landscape">
    <img src="../art/abstract.jpg" alt="Abstract Art" />
    <p>Title: It Doesn’t Always Have To Make Sense!<br/>Medium: Acrylic</p>
  </div>
  <div class="art-item smaller-dream">
    <img src="../art/Dream.jpg" alt="Dream" />
    <p>Title: Dream<br/>Medium: Ink</p>
  </div>
  <div class="art-item landscape">
    <img src="../art/friends.jpg" alt="Friends" />
    <p>Title: Friends<br/>Medium: Acrylic</p>
  </div>
  <div class="art-item">
    <img src="../art/girl.jpg" alt="The Girl" />
    <p>Title: The Girl<br/>Medium: Acrylic</p>
  </div>
  <div class="art-item">
    <img src="../art/first.jpg" alt="The First" />
    <p>Title: The First<br/>Medium: Charcoal</p>
  </div>
  <div class="art-item landscape">
    <img src="../art/walk.jpg" alt="Walk" />
    <p>Title: Walk<br/>Medium: Acrylic</p>
  </div>
</div>
