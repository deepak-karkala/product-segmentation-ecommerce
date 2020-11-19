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



// http://jsfiddle.net/2kev63mq/1/
$("#show_result").click(function (e) {

    //height and width of the container
    //var height = $("#pcVideo").height();
    //var width = $("#pcVideo").width();

    //get click position inside container
    var relativeX = e.pageX - this.offsetLeft;
    var relativeY = e.pageY - this.offsetTop;

    //$('#overlay_image').css("left", relativeX - 25);
    //$('#overlay_image').css("top", relativeY - 25);
    $('#overlay_image').css("left", e.pageX);
    $('#overlay_image').css("top", e.pageY);
    $('#overlay_image').show();
    //overlay
    var video = document.getElementById('videoElement');

    //video.pause();
});