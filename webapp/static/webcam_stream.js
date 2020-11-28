// https://www.kirupa.com/html5/accessing_your_webcam_in_html5.htm
var video = document.querySelector("#videoElement");

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}

var tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip1")
        .style("visibility", "hidden");


// http://jsfiddle.net/2kev63mq/1/
$("#videoElement").click(function (e) {

	var videoWidth = d3.select("#videoElement").node().offsetWidth;
	var videoHeight = d3.select("#videoElement").node().offsetHeight;
	var videoLeft = $("#videoElement").offset().left;
	var videoTop = $("#videoElement").offset().top;
	var videoBottom = videoTop + videoHeight;
	var videoRight = videoLeft + videoWidth;
	var videoCenterOffset = videoLeft + videoWidth*0.50;

	if (e.pageX >= videoCenterOffset) {
		return tooltip.html(`<img class='img_overlaid' src='static/product_segmented_image/product_segmented_image.png'>`)
				.style("top", Math.min(videoBottom-150, (e.pageY)) + "px")
				.style("left", Math.max(videoLeft, (e.pageX-100)) + "px")
				.style("visibility", "visible");
	} else {
		return tooltip.html(`<img class='img_overlaid' src='static/product_segmented_image/product_segmented_image.png'>`)
				.style("top", Math.min(videoBottom-150,(e.pageY)) + "px")
				.style("left", Math.max(videoLeft, (e.pageX-50)) + "px")
				.style("visibility", "visible");
	}

/*
    //height and width of the container
    //var height = $("#pcVideo").height();
    //var width = $("#pcVideo").width();

    //get click position inside container
    var relativeX = e.pageX - this.offsetLeft;
    var relativeY = e.pageY - this.offsetTop;

    console.log(e.pageX);


    //$('#overlay_image').css("left", relativeX - 25);
    //$('#overlay_image').css("top", relativeY - 25);
    $('#overlay_image').css("left", (e.pageX));
    $('#overlay_image').css("top", (e.pageY));
    //$('#overlay_image').css("display", "flex");
    $('#overlay_image').show();
    //overlay
    var video = document.getElementById('videoElement');

    //video.pause();
*/
});


set_size_slider();
function set_size_slider() {
	//var handle = $( "#custom-handle" );
	$( "#slider-range-min" ).slider({
	  create: function() {
	  },
	  slide: function( event, ui ) {
	  	$(".img_overlaid").css("width", ui.value+"vw").css("height", ui.value+"vw");
	   	//tooltip.style("height", ui.value+"vh").style("width", 200);
	  },
	  change: function(event, ui) {
	  	$(".img_overlaid").css("width", ui.value+"vw").css("height", ui.value+"vw");
	   	//tooltip.style("height", ui.value+"vh").style("width", 200);
	  },
	  range: "min",
	  max: 20,
	  min: 5,
	  step: 5,
	  value: 30,
	});
}
