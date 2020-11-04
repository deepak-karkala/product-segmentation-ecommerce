
function get_all_values_and_submit_form() {
	// Get value of currently filtered product
	//filter_product_category = document.getElementById("filter_product_button").value;
	filter_product_category = document.getElementsByClassName("button active")[0].value;
	// Add filtered product category as hidden input field before submitting form
	var input = $("<input>")
               .attr("type", "hidden")
               .attr("name", "filter_product_button").val(filter_product_category);
    $('#input_form').append($(input));


    /*
    selected_image_id = document.getElementById("product_radio").value;
    var input = $("<input>")
               .attr("type", "hidden")
               .attr("name", "product_radio").val(selected_image_id);
    $('#input_form').append($(input));
	*/

    $('#input_form').submit();
    //$(this).closest("form").submit();

    /*
    //var data = [];
    var data = $('#input_form').serializeArray();
    data.push({name: 'filter_product_button', value: filter_product_category});
    //data.push({name: 'product_radio', value: selected_image_id});

    $.ajax({
	  type: "POST",
	  url: "/",
	  data: data,
	});
	*/
}


// Enable form submit on radio button click
$('input[type=radio]').on('change', function() {
	get_all_values_and_submit_form();
    //$(this).closest("form").submit();
});



// Enable form submit on filter button click
//setupButtons();
function setupButtons() {
  d3.select('#toolbar')
    .selectAll('.button')
    .on('click', function () {
      // Remove active class from all buttons
      d3.selectAll('.button').classed('active', false);
      // Find the button just clicked
      var button = d3.select(this);

      // Set it as the active button
      button.classed('active', true);

      // Get the id of the button
      var buttonId = button.attr('id');
      console.log(buttonId);

	  //$(this).closest("form").submit();
      
    });
}