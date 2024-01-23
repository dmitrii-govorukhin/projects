date_start = '';
date_finish = '';

$(document).ready(function() {
    var table = $('#events').DataTable( {
        ajax: {
            url: 'ajax/event_history_table.php',
            type: 'POST'
        },
        processing: true,
        serverSide: true,
        deferRender: true,
        order: [[ 1, 'desc' ]],
        colReorder: true,
        pageLength: 10,
        lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
        columns: [
            { data:           null,
                className:      'details-control',
                orderable:      false,
                defaultContent: ''
            },
            { data: "WRITETIME",
                render: function ( data, type, row ) {
                    var DateTimeSplit = data.split(' ');
                    var DateSplit = DateTimeSplit[0].split('-');
                    return type === "display" ?
                        DateSplit[2] + '.' + DateSplit[1] + '.' + DateSplit[0] + ' ' + DateTimeSplit[1]:
                        data;
                },
                orderable: true
            },
            { data: "FIRST_OCCURRENCE",
                render: function ( data, type, row ) {
                    var DateTimeSplit = data.split(' ');
                    var DateSplit = DateTimeSplit[0].split('-');
                    return type === "display" ?
                        DateSplit[2] + '.' + DateSplit[1] + '.' + DateSplit[0] + ' ' + DateTimeSplit[1]:
                        data;
                },
                orderable: true
            },
            { data: "SERIAL",
                className: 'dt-body-center',
                render: function ( data, type, row ) {
                    return type === "display" ?
                        data + '<img class="click" id="serial" src="images/filter.png" hspace="10" align="top">' :
                        data;
                },
                orderable: false
            },
            { data: "PFR_TORG",
                orderable: false
            },
            { data: "NODE",
                orderable: false
            },
            { data: "PFR_OBJECT",
                render: function ( data, type, row ) {
                    return (type === "display" && data.length > 0) ?
                        data + '<a href="http://10.10.10.10/SCCD_trigger.php?ServiceName=' + data + '" target="_blank"' +
                        '><img src="images/link.png" align="top" hspace="5"></a>' :
                        data;
                },
                orderable: false
            },
            { data: "PFR_KE_TORS",
                render: function ( data, type, row ) {
                    return (type === "display" && data.length > 0) ?
                        '<a href=\'http://' + SCCD_server + '/maximo/ui/login?event=loadapp&value=CI&additionalevent=useqbe&additionaleventvalue=CINAME=' + data + '\' ' +
                        'target=\'blank\'>' + data + '</a>' +
                        '<a href="http://10.10.10.10/SCCD_trigger.php?KE=' + data + '" target="_blank"' +
                        '><img src="images/link.png" align="top" hspace="5"></a>' :
                        data;
                },
                orderable: false
            },
            { data: "PFR_SIT_NAME",
                render: function ( data, type, row ) {
                    return type === "display" ?
                        '<table width="100%" style="background:transparent"><tr><td>' + data + '</td>' + '<td id="chart" align="right"></td></tr></table>' :
                        data;
                },
                orderable: false
            },
            { data: "SEVERITY",
                className: 'cell_severity',
                orderable: false
            },
            { data: "TTNUMBER",
                render: function ( data, type, row ) {
                    return type === "display" ?
                        '<a href=\'http://' + SCCD_server + '/maximo/ui/maximo.jsp?event=loadapp&value=incident&additionalevent=useqbe&additionaleventvalue=ticketid=' + data +
                            '&datasource=NCOMS\' target=\'blank\'>' + data + '</a>' :
                        data;
                },
                className: 'dt-body-center',
                orderable: false
            },
            { data: "PFR_TSRM_CLASS",
                className: 'cell_tsrm_class',
                orderable: false
            },
            { data: "CLASSIFICATIONID",
                render: function ( data, type, row ) {
                    return (type === "display" && data !== null) ?
                        '<a href=\'http://' + SCCD_server + '/maximo/ui/login?event=loadapp&value=ASSETCAT&additionalevent=useqbe&additionaleventvalue=CLASSIFICATIONID=' + data + '\' ' +
                        'target=\'blank\'>' + data + '</a>' :
                        data;
                },
                className: 'dt-body-center',
                orderable: false
            },
            { data: "CLASSIFICATIONGROUP",
                orderable: false
            },
            { data: "PFR_TSRM_WORDER",
                render: function ( data, type, row ) {
                    return type === "display" && data != null ?
                        '<a href=\'http://' + SCCD_server + '/maximo/ui/?event=loadapp&amp;value=wotrack&amp;additionalevent=useqbe&amp;additionaleventvalue=wonum=' + data +
                            '&amp;forcereload=true\' target=\'blank\'>' + data + '</a>' :
                        data;
                },
                className: 'dt-body-center',
                orderable: false
            }
        ],
        fnRowCallback: function(nRow, aData) {
            if (aData['SAMPLED_SIT']) {
                if (aData['SIT_IN_COLLECTION'])
                    $('td#chart', nRow).html('<a id="cell_pfr_sit_name" href=\'#\'><img src="images/chart.png" align="top" hspace="10" width="24" height="24"></a>');
                else
                    $('td#chart', nRow).html('<img src="images/chart_inactive.png" align="top" hspace="10" width="24" height="24">');
            }

            $('a#cell_pfr_sit_name', nRow).attr('onclick', 'showGraph_operative(' + aData['SERIAL'] + '); return false;');

            switch (aData["SEVERITY"]) {
                case "Critical":
                    $('td.cell_severity', nRow).attr('class', 'red_status dt-body-center'); break;
                case "Marginal":
                case "Minor":
                case "Warning":
                    $('td.cell_severity', nRow).attr('class', 'yellow_status dt-body-center'); break;
                case "Informational":
                    $('td.cell_severity', nRow).attr('class', 'blue_status dt-body-center'); break;
                case "Harmless":
                    $('td.cell_severity', nRow).attr('class', 'green_status dt-body-center'); break;
                default:
                    break;
            }

            if (aData["PFR_TSRM_CLASS"].indexOf('Off') == 0 || aData["PFR_TSRM_CLASS"].indexOf('Test') == 0)
                    $('td.cell_tsrm_class', nRow).attr('class', 'blue_status');
        },
        drawCallback: function(settings) {
            // filters and data for export to excel
            var api = this.api();
            $('a#excel').attr('href', 'http://10.10.10.10/event_history_excel.php?data=' + JSON.stringify(api.ajax.json(), ['options']));
        },
        initComplete: function (settings, json) {
            // lists filters
            this.api().columns().every( function () {
                var column = this;
                var select = $('select', this.footer());
                if (column.dataSrc() == 'SEVERITY') {
                    select.append('<option value="5">Critical</option>');
                    select.append('<option value="4">Marginal</option>');
                    select.append('<option value="3">Minor</option>');
                    select.append('<option value="2">Warning</option>');
                    select.append('<option value="1">Informational</option>');
                    select.append('<option value="0">Harmless</option>');
                }

                select.on('change', function () {
                    var val = $(this).val();
                    column
                        .search(val)
                        .draw();
                });
            } );
        },
        dom: 'lr<"rightimg"B><"rightimg"f>rtip',
        buttons: [
            'colvis', 'copy', 'print'
        ],
    } );

    $('#events tfoot tr').appendTo('#events thead');

    table.columns().every( function () {
        var that = this;

        if ($('input', this.footer()).val()) {
            if ($('input', this.footer()).val().length > 0) {
                var att = $('input', this.footer()).attr('id');
                if (typeof att !== typeof undefined && att !== false && att == 'ptk')
                    table.search($('input', this.footer()).val());
                else
                    this.search('^' + $('input', this.footer()).val());
                this.draw();
            }
        }

        $( 'input', this.footer() ).on( 'keyup change', function () {
            if ( that.search() !== this.value ) {
                var attr = $(this).attr('id');
                if (attr == 'start') {
                    date_start = this.value;
                    that.search(date_start + '*' + date_finish);
                }
                else if (attr == 'finish') {
                    date_finish = this.value;
                    that.search(date_start + '*' + date_finish);
                }
                else
                    that.search(this.value);
                that.draw();
            }
        } );
    } );

    $('#events tbody').on('click', '#serial', function () {
        var td = $(this).closest('td');
        var cell = table.cell( td );
        var serial = cell.data();

        var column = table.column(3);
        var input = $('input', column.footer());
        var search = input.val() == serial ? '' : serial;

        input.val(search);
        column
            .search(search)
            .draw();
    } );

    // Add event listener for opening and closing details
    $('#events tbody').on('click', 'td.details-control', function () {
        var tr = $(this).closest('tr');
        var row = table.row( tr );

        if ( row.child.isShown() ) {
            row.child.hide();
            tr.removeClass('shown');
        }
        else {
            row.child( format(row.data()) ).show();
            tr.addClass('shown');
        }
    } );

    // Add event listener for toggle all details
    $('img#details').on( 'click', function () {
        table.rows().each(function () {
            $('.details-control').trigger( 'click' );

            if ($('img#details').attr('src') == 'images/details_close.png') {
                $('img#details').attr('src', 'images/details_open.png');
                $('img#details').attr('title', 'Unfold all parts');
            }
            else {
                $('img#details').attr('src', 'images/details_close.png');
                $('img#details').attr('title', 'Fold all parts');
            }
        } );
    } );

    function format ( d ) {
        return '<table cellpadding="0" cellspacing="10" border="0" style="padding-left:50px;">'+
                '<tr>'+
                    '<td>Description:</td>'+
                    '<td>'+
                        d.DESCRIPTION +
                    '</td>'+
                '</tr>'+
                '<tr>'+
                    '<td valign="top">' + (d.TRACEROUTE_TITLE.trim() == '' ? '' : 'Tracert result:<br>' + d.TRACEROUTE_TITLE) + '</td>'+
                    '<td>'+
                        d.TRACEROUTE_DATA +
                    '</td>'+
                '</tr>'+
            '</table>';
    }
} );
