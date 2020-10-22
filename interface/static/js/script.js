$(function(){

    var blank_choice = 
    "<div class='d-flex justify-content-between choice row'></div>"
    var blank_word = "<div class='btn word col-8'></div>"
    var yes_button = "<i class='fa fa-check fa-2x yes col-2'></i>"
    var no_button = "<i class='fa fa-times fa-2x no col-2'></i>"
    
    // hide loader
    $("#load > p").hide()

    // dynamically add choices
    var add_choice = function(cluster, text, language) {
        // find all text values in cluster
        var all_text = $.map($(cluster).find('.word'), $.text);
        var query = $(".query").text()
        // make sure word is not in cluster already
        if ($.inArray(text, all_text) == -1 && text != query) {
            var new_choice = $(blank_choice).clone();
            var new_word = $(blank_word).clone();
            new_word.text(text);
            new_choice.append(new_word);
            new_choice.append(yes_button);
            new_choice.append(no_button);
            new_choice.addClass(language);
            new_choice.addClass('pos');
            new_word.addClass(language);
            new_word.addClass('new');
            new_word.addClass('pos');
            cluster.append(new_choice);
        }
    }

    // input for l1
    $(".form-control.l1").autocomplete({
        minLength: 1,
        delay: 500,
        // callback to get word choices 
        source: function(request, response) {
            $("#load > p").show()

            $.get("/autocomplete/1", {
                query: request.term
            }, function(data) {
                $("#load > p").hide()

                var choices = data["choices"];
                response(choices);
            });
        },
        select: function(ev, ui) {
            var text = ui.item.label;
            var nn_cluster = $(".cluster-nn.l1");
            add_choice(nn_cluster, text, "l1");
            // clear textbox after select
            this.value = "";
            // cancel the event to prevent autocomplete update the field
            return false;  
        },
    });

    // input for l2
    $(".form-control.l2").autocomplete({
        minLength: 1,
        delay: 500,
        // callback to get word choices 
        source: function(request, response) {
            $("#load > p").show()

            $.get("/autocomplete/2", {
                query: request.term
            }, function(data) {
            $("#load > p").hide()

                var choices = data["choices"];
                response(choices);
            });
        },
        select: function(ev, ui) {
            var text = ui.item.label;
            var nn_cluster = $(".cluster-nn.l2");
            add_choice(nn_cluster, text, "l2");
            // clear textbox after select
            this.value = "";
            // cancel the event to prevent autocomplete update the field
            return false;  
        }
    });



    // context
    var context = function (ev) {
        $("#load > p").show()

        var query = $(ev.target).text()
        if ($(ev.target).hasClass("l1")) {
            var lang = "1"
        } else {
            var lang = "2"
        }


        $.get("/context/"+lang, {query: query})
            .done(function(data) {
                $("#load > p").hide()

                var doc = data['doc'];
                // pop-up
                var title = $("#context .modal-title");
                title.text(query) 
                var body = $("#context .modal-body");
                body.html(doc)
                $("#context").modal();
            });
        
    };

    $("#clusters").on("click", ".query, .word", context)

    // yes/no buttons
    var choice = function (ev) {
        var word = $(ev.target).siblings(".word")
        var choice = $(ev.target).parent()
        if ($(ev.target).hasClass("yes")) {
            if (word.hasClass("pos")) {
                word.removeClass("pos");
                choice.removeClass("pos");

            } else {
                word.removeClass("neg");
                choice.removeClass("neg");
                word.addClass("pos");
                choice.addClass("pos");

            }

            // don't let click be processed to other clause
            ev.stopPropagation();

        } else if ($(ev.target).hasClass("no")) {
            if (word.hasClass("neg")) {
                word.removeClass("neg");
                choice.removeClass("neg");

            } else {
                word.removeClass("pos");
                choice.removeClass("pos");
                word.addClass("neg");
                choice.addClass("neg");

            }

            

            // don't let click be processed to other clause
            ev.stopPropagation();
        } 
        else {
            return false;
        }
    };

    $("#clusters").on("click", ".yes, .no, .word", choice)


    function collectText(selector) {
        var array = $(selector).map(function (){
            return $(this).text()
        }).get();
        return array;
    }

    // save progress and move on to next page
    var save = function(ev) {
        var pos1 = collectText(".word.pos.l1");
        var pos2 = collectText(".word.pos.l2");
        var neg1 = collectText(".word.neg.l1");
        var neg2 = collectText(".word.neg.l2");
        var new1 = collectText(".word.new.l1");
        var new2 = collectText(".word.new.l2");

        var data = {
            "pos1":pos1,
            "pos2":pos2,
            "neg1":neg1,
            "neg2":neg2,
            "new1":new1,
            "new2":new2,
        };

        var page = $("#current").text();
        var entry = String(Number(page)-1);
        var total = $("#total").text();

        // save in database

        $.ajax({
            url: "/save/"+entry,
            contentType: 'application/json',
            data: JSON.stringify(data),
            type: 'POST',
            success: function(result) {
                // go to next page
                if (Number(page) >= Number(total)){
                    // finish if no more pages left
                    window.location.href = "/finish"
                } else {
                    window.location.href = "/ui/"+page;
                };

            }

        });

    }
    $("#bar").on("click", "#save", save)

    // instructions
    


});
