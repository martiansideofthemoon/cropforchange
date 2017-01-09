distribution = 
[[ 0.236257  ,  0.04692218,  0.20064587,  0.27006631,  0.24610865],
 [ 0.25956452,  0.20037058,  0.11330117,  0.18309477,  0.24366896],
 [ 0.03462285,  0.12199772,  0.30176772,  0.30892052,  0.23269119],
 [ 0.14768843,  0.43006252,  0.22314986,  0.10070357,  0.09839562],
 [ 0.13476074,  0.20325505,  0.21877775,  0.28007741,  0.16312904],
 [ 0.05843576,  0.09840608,  0.06965531,  0.40963047,  0.36387238],
 [ 0.04799431,  0.28932893,  0.15560844,  0.35926815,  0.14780017],
 [ 0.33762325,  0.1006453 ,  0.25885253,  0.25791053,  0.04496839],
 [ 0.29736977,  0.38960393,  0.0143184 ,  0.13547417,  0.16323372],
 [ 0.28743664,  0.06803946,  0.22860892,  0.00475499,  0.41115998],
 [ 0.30387412,  0.21161437,  0.05843983,  0.13943878,  0.28663289],
 [ 0.22890264,  0.41650632,  0.20146101,  0.10185317,  0.05127686],
 [ 0.28142904,  0.06912253,  0.29146848,  0.17506701,  0.18291293],
 [ 0.15229192,  0.23234493,  0.22887555,  0.14250672,  0.24398087],
 [ 0.27450447,  0.1160405 ,  0.07602911,  0.2687476 ,  0.26467832],
 [ 0.36545777,  0.09747976,  0.44755701,  0.07444514,  0.01506031]]


window.onload = function(){
    var canvas = document.getElementById("canvas"),
        context = canvas.getContext("2d"),
        width = canvas.width = 640,
        height = canvas.height = 480;

    //horizontal grid
    for(var i = 0;i<=4;i+=1){
        context.beginPath();
        context.moveTo(0, i*height/4.0);
        context.lineTo(width, i*height/4.0);
        context.stroke();
    }

    //vertical grid
    for(var i =0;i<=4;i+=1){
        context.beginPath();
        context.moveTo(i*width/4.0,0);
        context.lineTo(i*width/4.0,height);
        context.stroke();
    }
    cellw = width/4.0;
    cellh = height/4.0; 
    for(var i = 0;i<4;i+=1){
        for(var j = 0;j<4;j+=1){
            startw = i*cellw+1;
            context.fillStyle = "rgb(200,60,80)";
            context.fillRect(startw+1, j*cellh+cellh/3.0, distribution[i*4+j][0]*cellw-1, cellh/3.0);
            startw += distribution[i*4+j][0]*cellw;
            context.fillStyle = "rgb(40,240,80)";
            context.fillRect(startw+1, j*cellh+cellh/3.0, distribution[i*4+j][1]*cellw-1, cellh/3.0);
            startw += distribution[i*4+j][1]*cellw;
            context.fillStyle = "rgb(230,0,80)";
            context.fillRect(startw+1, j*cellh+cellh/3.0, distribution[i*4+j][2]*cellw-1, cellh/3.0);
            startw += distribution[i*4+j][2]*cellw;
            context.fillStyle = "rgb(200,123,42)";
            context.fillRect(startw+1, j*cellh+cellh/3.0, distribution[i*4+j][3]*cellw-1, cellh/3.0);
            startw += distribution[i*4+j][3]*cellw;
            context.fillStyle = "rgb(80,60,200)";
            context.fillRect(startw+1, j*cellh+cellh/3.0, distribution[i*4+j][4]*cellw-1, cellh/3.0);
        }
    } 
}

$(document).ready(function(){
    $("#add").click(function(){
        $("#overlay").show();
        $("#add_form").show();
    });
    $("#play").click(function(){
        $("#overlay").show();
        $("#play_form").show();
    });
    $("#close_overlay").click(function(){
        $("#add_form").hide();
        $("#play_form").hide();
        $("#overlay").hide();
    });
    $("#play_form").submit(function(e){
        e.preventDefault();
        var crop_vector = [this.wheat.value,
                           this.barley.value,
                           this.paddy.value,
                           this.mustard.value,
                           this.corn.value];
        $.ajax({
            url:"execute.php",
            method:"GET",
            data:{"crop":crop_vector}
        }).done(function(result){
            alert(result);
        });
    });
});
