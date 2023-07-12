// -------- List of publication --------


query = "q=author%3AJuillard%2C+Sandrine&fl=title%2C+author%2C+bibcode%2C+year&rows=6";

$(document).ready(function() {
    $.ajax({
        url: 'https://corsproxy.io/?https://api.adsabs.harvard.edu/v1/search/query?'+query,
        beforeSend: function(xhr) {
             xhr.setRequestHeader("Authorization","Bearer MPO6caNhGOjkCeT7ebkh8GyksYnbOeDfzb0hpiZk");
        }, 
        dataType: "text",       
        headers: {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS', 'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token'},
        success: function(data){
            
            var json = JSON.parse(data)['response']['docs'];
            var info_type = ["title", "year", "author"];
            for (var i=0;i<json.length;++i)
            {
                var publication_i = document.createElement("li");
                for (var k=0;k<info_type.length;++k){
                
                    // Content from JSON
                    var info = info_type[k]
                    var content = json[i][info]
                    
                    if (info == "title"){
                        content = content[0];
                        var adress = "https://ui.adsabs.harvard.edu/abs/"+json[i]['bibcode']+"/abstract";
                    }
                    if (info == "author"){
                        var tmp_content = content.splice(0, 3);
                        if(content.length > 3){
                            tmp_content.push("et al.");
                        }
                        tmp_content.join(',    ');
                        content = tmp_content;
                    }
                    
                    // Create div
                    if (info == "title"){
                        var ele = document.createElement("a");
                        ele.setAttribute("href", adress);
                    }else {
                        var ele = document.createElement("div");
                    }
                    ele.classList.add(info);
                    ele.appendChild(document.createTextNode(content));
                    publication_i.appendChild(ele);
                      
                    document.getElementById("ADS").appendChild(publication_i);
                    }
            }

        },
    })
});
    

// -------- Line under titles --------

$(document).ready(function() {
    
    // Get max width
    var max = 0;
    var width = 0
    for (var i = 0; i < document.getElementsByClassName("sec").length; i++) {
    
        width = document.getElementsByClassName("sec")[i].clientWidth
        if (width*1 > max*1){ max = width;}
    };
    
    // Set width to all bars
    for (var i = 0; i < document.getElementsByClassName("rounded").length; i++) {
        document.getElementsByClassName("rounded")[i].style.width = (max+50)+'px';
    };

    
});

// -------- Srcoll of images --------

var scrollHandler = null;
  
function autoScroll () {
    clearInterval(scrollHandler);
    scrollHandler = setInterval(function() {
      var nextScroll = document.getElementById("public").scrollLeft += 1;
    },1);
}

 function handleOnScroll () {
  clearInterval(scrollHandler);
  setTimeout(autoScroll, 200);
 };
 
 autoScroll();