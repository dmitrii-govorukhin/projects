<?php
/*
	by GDV
	2015-2019 RedSys
*/
header('Content-Type: text/html;charset=UTF-8');
?>
<!DOCTYPE html>
<html>
<head>
    <META content="text/html; charset=UTF-8" http-equiv="Content-Type">
	<link href="css/style.css" type="text/css" rel="stylesheet">
    <title>Настройка интеграции с СТП</title>
	<script src="scripts/jquery-3.2.1.min.js"></script>
    <script src="scripts/common.js"></script>
    <script src="scripts/SCCD_trigger.js"></script>
</head>
<body>
	<?php
	// common functions
    require_once 'connections/TBSM.php';
    require_once 'connections/MAXDB76.php';
	include 'functions/utime.php';
	include 'functions/regions.php';	
	include 'functions/user_roles.php';
    include 'functions/remote_exec.php';
    include 'functions/tbsm.php';
    include 'functions/PFR_LOCATIONS_record_form.php';

    define ("MAX_CHILD_SERVICES",           500);
    define ("MIN_STRINGS_TO_HIDE_TABLE",     50);

    define ("TO_WORK_MODE",           "Перевести в рабочий режим");
    define ("TO_MAINTENANCE_MODE",    "Перевести в режим обслуживания");
    define ("TO_BE_NOT_BLUE",         "Из синего цвета");
    define ("TO_BE_BLUE",             "В синий цвет");

    $pfr_ke_tors = array ();	// PFR_KE_TORS, SERVICE_NAME and PFR_ID_TORG fields from PFR_LOCATIONS table
    $array_sources = [ ];	    // data sources array
    $count_services = 1;        // number of services
    $table_N1_titles = array (
        "Код индикатора",
        "Тип агента",
        "Расположение агента",
        "Сервер объектов",
        "Сервер TEMS",
        "Регион",
        "Журнал событий",
        "Узел",
        "Объект мониторинга",
        "Ссылка на КЭ",
        "Фильтр по ситуациям",
        "Назначение",
        "URL",
        "Отправка инцидентов",
        "Строка в PFR_LOCATIONS",
    );
    $table_N1_data = array ();  // data array for the first form table

    $table_N1_1_data = array ();  // data array for the AEL table

    $table_N2_titles = array (
        "Расположение агента в сервисной модели мониторинга",
        "Ссылка на КЭ*",
        "URL",
        "Статус ситуации в TEMS",
        "Наименование ситуации в TEMS (фильтр ситуации, если применимо)",
        "Код события",
        "Описание**",
        "Формула ситуации в TEMS",
        "Критичность",
        "Частота опроса (ч:мин:с) и кол-во последовательных проверок",
        "Генерация тестового события" ); // titles for the second form table
    $table_N2_data = array (array ( 
        "number" => "",
        "place" => "",
        "ke" => "",
        "url" => "",
        "hubs" => array(array()),
        "sit_name" => "",
        "sit_code" => "",
        "descr" => "",
        "sit_form" => "",
        "severity" => "",
        "period" => "",
        "unique" => "",
        )); // data array for the second form table

    $table_N3_titles = array (
        "Имя конфигурационного элемента",
        "Статус",
        "РЗ",
        "Код события",
        "Имя ситуации",
        "Код классификации",
        "Пауза при создании РЗ (мин)",
        "Рабочая группа",
        "Руководитель",
        "Регион в мониторинге",
        "Регион в СТП",
    );


    $table_N3_data = array ();      // data array for the third form table

	$event_codes = array ( "PFR_TSPC_EIF_EVENT" => "89200",
						   "PFR_SD_EIF_EVENT" => "87350" ); // test event codes for special situations

    $incident_write = ""; 				// incident exists
    $all_inc_send_change = 0;           // change send status of all incidents
    $all_inc_send_on = null;            // incidents send status is "all on"
    $all_inc_send_off = null;           // incidents send status is "all off"
    $run_event = "";                    // run test event for the situation(s)
    $err_num = 0;                       // error number for test events
    $output = $connection_SCCD ? "" : "Отсутствует подключение к БД СТП!"; 						// top header informational message
    $log_flag = true;                   // to save in log or not to save

	$log_file = 'logs/SCCD_trigger.log'; // log file
	$sql_file = 'SCCD_trigger.sql'; 	// sql command file
    $log_file_sql = 'logs/SCCD_sql.log'; // log file of sql queries
	$report_busy_file = 'SCCD_trigger.rep'; // report update busy by another user info file
    $undeleted = 'SCCD_trigger.sav';    // undeleted services with several parents
    $log_regions = 'logs/SCCD_regions_check.log';   // log with situations download problem regions

    $pass_URL = '/pfr_other/Passports/';                // passports web path
	$pass_dir = '/usr/local/apache2/htdocs'.$pass_URL;	// passports directory
	$pass_arch_dir = $pass_dir.'Archive/';				// archived passports directory
	$surv_dir = '/usr/local/apache2/htdocs/pfr_other/Surveys/';	// surveys directory
	$surv_arch_dir = $surv_dir.'Archive/';						// archived surveys directory
	$output_pass = ""; 					// passport upload informational message

    $initiator = '';            // name of incident initiator
	$command = "";				// sql command for test events
	$q_status = true;			// status of sql command run
	$event_lifetime = 600;		// test event lifetime in seconds
	$event_lifetime_recommended = 0; // recommended test event lifetime in seconds
    $max_delay = 0;             // maximum delay for РЗ

	$template = ""; 			// selected service's template
	$multi_scheme = false;	 	// selected service is container not endpoint service
    $light_scheme = false;      // light form output for container
	$level = 0;					// iteration level
	$path_history = [ ];		// array for history tree from parents to childs
	$results = [ ];				// array for endpoint childs
	$services = [ ];			// sorted and unique array for endpoint childs
    $snapshot = '';             // incident send snapshot presense
    $snapshot_action = '';      // action with incident send snapshot

	$arr_BigFix = [ ];			// array for BigFix check command output
	$is_BigFix_agent = '';		// array BigFix presence

	// *********************************************************** WEB FORM PREPARE & HANDLE ***********************************************************

	// get script parameters
	$NODEID = isset($_GET["ServiceName"]) ? $_GET["ServiceName"] : "";
    $KE = isset($_GET["KE"]) ? strtoupper($_GET["KE"]) : "";

    // KE parameter was sent to script
    if (!empty($KE)) {
        $sel = "SELECT SERVICEINSTANCENAME, DISPLAYNAME FROM TBSMBASE.SERVICEINSTANCE WHERE SERVICEINSTANCENAME IN 
                (select SERVICE_NAME from DB2INST1.PFR_LOCATIONS where PFR_KE_TORS = '$KE')";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);

        while ($row = db2_fetch_assoc($stmt))
            $results[] = $row['SERVICEINSTANCENAME']." (".$row['DISPLAYNAME'].")";
        $service_arr = array_unique($results);
        sort($service_arr);
        if (empty($service_arr))
            exit("Для КЭ = <b>".$KE."</b> не найдено ни одного сервиса!");

        echo "Для КЭ = <b>".$KE."</b> найдены следующие сервисы:<ul>";
        foreach ($service_arr as $s) {
            list($inst, $displ) = explode(' (', $s);
            echo "<li><a href=\"http://10.103.0.60/pfr_other/SCCD_trigger.php?ServiceName=" . $inst . "\" >" . $s . "</a>";
        }
        echo "</ul>Для перехода в форму \"Настройка интеграции с СТП\" кликните по нужному сервису...";
        exit();
    }

    // existence and template of the service
    $sel = "SELECT SERVICEINSTANCEID, SERVICESLANAME, DISPLAYNAME, TIMEWINDOWNAME FROM TBSMBASE.SERVICEINSTANCE WHERE SERVICEINSTANCENAME = '$NODEID'";
    $stmt = db2_prepare($connection_TBSM, $sel);
    $result = db2_execute($stmt);
    $row = db2_fetch_assoc($stmt);
    if (empty($row['SERVICEINSTANCEID']))
        exit('Wrong ServiceName parameter!');
    $template = $row['SERVICESLANAME'];
    $maint_status = $row['TIMEWINDOWNAME'];
    $NODEDESCRIPTION = $row['DISPLAYNAME'];

    // test event lifetime in seconds
    if (isset($_POST['formId']['event_lifetime'])) 	{				// event lifetime was just entered
        $event_lifetime = $_POST['formId']['event_lifetime'];
        setcookie('event_lifetime', $event_lifetime, time() + 1800);
    }
    else if (isset($_COOKIE["event_lifetime"])) {			// event lifetime was entered earlier
        $event_lifetime = $_COOKIE["event_lifetime"];
    }

    // output buffering turn on
    if (ob_get_level() == 0)
        ob_start();
    $startMemory = memory_get_usage();
    echo "<p id='to_remove'><font color='red'>Пожалуйста, подождите (Esc - прервать)</font><img src=\"images/inprogress.gif\" hspace=10><br><br>";
    ob_flush();
    flush();

    // is it endpoint service or container?
    $sel = "SELECT COUNT(*) as CNT
            FROM TBSMBASE.SERVICEINSTANCE, TBSMBASE.SERVICEINSTANCERELATIONSHIP
            WHERE PARENTINSTANCEKEY = (SELECT SERVICEINSTANCEID
                                       FROM TBSMBASE.SERVICEINSTANCE
                                       WHERE SERVICEINSTANCENAME = '$NODEID')
            AND SERVICEINSTANCEID = CHILDINSTANCEKEY";
    $stmt = db2_prepare($connection_TBSM, $sel);
    $result = db2_execute($stmt);
    $row = db2_fetch_assoc($stmt);

    // it's container
    if ($row['CNT'] > 0 and $template != 'HYPERVISOR_CLUSTER:Standard' and $template != 'CLUSTER_OF_SERVERS:Standard') {
        echo "Выбор дочерних сервисов...&emsp;";
        ob_flush();
        flush();

        // find all child services
        $sel = "SELECT SERVICEINSTANCEID FROM TBSMBASE.SERVICEINSTANCE WHERE SERVICEINSTANCENAME = '$NODEID'";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);
        $row = db2_fetch_assoc($stmt);

        // recursive function call
        ext_tree($row['SERVICEINSTANCEID'], $connection_TBSM, $level, $path_history);

        // fill array of services
        foreach ($results as $val)
            if (!array_key_exists($val['service'], $services))
                $services[$val['service']] = array(
                    'displayname' => $val['display'],
                    'inc_change' => '',
                );
        ksort($services);
        $multi_scheme = true;

        $count_services = count($services);
        echo $count_services . "<img src='images/ok.png' hspace='10'><br>";

        // too many child services
        if ($count_services > MAX_CHILD_SERVICES) {
            ?><script>alert("Количество сервисов в выбранной подсистеме более <?php echo MAX_CHILD_SERVICES; ?>.\n\r" +
                    "Будет выведена упрощённая версия формы (без таблиц)...");</script><?php
            $light_scheme = true;
        }
    }
    // it's endpoint service
    else
        $services[$NODEID] = array(
            'displayname' => $NODEDESCRIPTION,
            'inc_change' => '',
        );

    // incident send snapshot presence
    if ($multi_scheme) {
        $sel = "select distinct TIMESTAMP from DB2INST1.PFR_MAINTENANCE_SNAPSHOTS where SERVICE_NAME = '$NODEID'";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);
        $row = db2_fetch_assoc($stmt);
        $snapshot = empty($row['TIMESTAMP']) ? '' : $row['TIMESTAMP'];
    }

    // passport web form button was pressed
    if ((isset($_POST['passport_rename']) or isset($_POST['protocol_rename']) or isset($_POST['data_edit']) or isset($_POST['passport_upload']) or isset($_POST['protocol_upload'])) and !empty($acs_user)) {
        echo "Обработка нажатия кнопки из раздела сведений о Паспортах...";
        ob_flush();
        flush();

        // passport code find
        $sel = "select FILE_CODE from DB2INST1.PFR_PASSPORT_SERVICES where SERVICE_NAME = '$NODEID'";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);
        $row = db2_fetch_assoc($stmt);
        $pass_code = $row['FILE_CODE'];

        // passport requisites find
        $sel = "select * from DB2INST1.PFR_PASSPORT_CODING where FILE_CODE = '$pass_code'";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);
        $row = db2_fetch_assoc($stmt);

        if (empty($pass_code) or empty($row))
            $output = "Паспорт для ".$NODEID." не заведён в систему!";
        else {
            // passport rename
            if (isset($_POST['passport_rename'])) {
                $new_name = $_POST['passport_name'];
                $sel = "update DB2INST1.PFR_PASSPORT_CODING set PASS_DISPLAY_NAME = '$new_name' where FILE_CODE = '$pass_code'";
                $stmt = db2_prepare($connection_TBSM, $sel);
                $result = db2_execute($stmt);
                $output = "Наименование Паспорта для ".$NODEID." изменено.";
            }

            // protocol rename
            if (isset($_POST['protocol_rename'])) {
                $new_name = $_POST['protocol_name'];
                $sel = "update DB2INST1.PFR_PASSPORT_CODING set PROC_DISPLAY_NAME = '$new_name' where FILE_CODE = '$pass_code'";
                $stmt = db2_prepare($connection_TBSM, $sel);
                $result = db2_execute($stmt);
                $output = "Наименование Протокола для ".$NODEID." изменено.";
            }

            // versions edit
            if (isset($_POST['data_edit'])) {
                $sel_ver = "select * from DB2INST1.PFR_PASSPORT_VERSIONS where FILE_CODE = '$pass_code'";
                $stmt_ver = db2_prepare($connection_TBSM, $sel_ver);
                $result_ver = db2_execute($stmt_ver);

                while ($row_ver = db2_fetch_assoc($stmt_ver)) {
                    // passport version changes
                    if (strcmp($_POST['passport_version'][$row_ver['ID']], $row_ver['PASS_VERSION']) or strcmp($_POST['passport_date'][$row_ver['ID']], $row_ver['PASS_DATE']) or isset($_POST['passport_signed'][$row_ver['ID']])) {
                        $sel_upd = "update DB2INST1.PFR_PASSPORT_VERSIONS set PASS_VERSION = '".$_POST['passport_version'][$row_ver['ID']]."', PASS_DATE = '".$_POST['passport_date'][$row_ver['ID']]."', PASS_SIGNED = ".(isset($_POST['passport_signed'][$row_ver['ID']]) ? "'Y'" : "PASS_SIGNED")." where ID = ".$row_ver['ID'];
                        $stmt_upd = db2_prepare($connection_TBSM, $sel_upd);
                        $result_upd = db2_execute($stmt_upd);

                        // passport file rename
                        if (strcmp($_POST['passport_version'][$row_ver['ID']], $row_ver['PASS_VERSION']) or strcmp($_POST['passport_date'][$row_ver['ID']], $row_ver['PASS_DATE'])) {
                            $old_file_name = "Паспорт_" . $row['PTK'] . "_" . $row['REGION'] . "_" . $row['ENVIRONMENT'] . "_версия_" . $row_ver['PASS_VERSION'] . "_от_" . $row_ver['PASS_DATE'] . "." . $row_ver['PASS_FILE'];
                            $new_file_name = "Паспорт_" . $row['PTK'] . "_" . $row['REGION'] . "_" . $row['ENVIRONMENT'] . "_версия_" . $_POST['passport_version'][$row_ver['ID']] . "_от_" . $_POST['passport_date'][$row_ver['ID']] . "." . $row_ver['PASS_FILE'];
                            if (file_exists($pass_dir . $old_file_name))
                                if (rename($pass_dir . $old_file_name, $pass_dir . $new_file_name))
                                    $output .= " Файл Паспорта \"" . $old_file_name . "\" переименован в \"" . $new_file_name . "\".";
                                else
                                    $output .= " Ошибка переименования Файла Паспорта \"" . $old_file_name . "\" в \"" . $new_file_name . "\"!";
                            else
                                $output .= " Файл Паспорта \"" . $old_file_name . "\" не найден на диске!";
                        }
                        else
                            $output .= " Отметка об утверждении Паспорта установлена.";
                    }

                    // protocol version changes
                    if ((isset($_POST['protocol_date'][$row['ID']]) and strcmp($_POST['protocol_date'][$row['ID']], $row['PROC_DATE'])) or isset($_POST['protocol_signed'][$row_ver['ID']])) {
                        $sel_upd = "update DB2INST1.PFR_PASSPORT_VERSIONS set PROC_DATE = '".$_POST['protocol_date'][$row_ver['ID']]."', PROC_SIGNED = ".(isset($_POST['protocol_signed'][$row_ver['ID']]) ? "'Y'" : "PROC_SIGNED")." where ID = ".$row_ver['ID'];
                        $stmt_upd = db2_prepare($connection_TBSM, $sel_upd);
                        $result_upd = db2_execute($stmt_upd);

                        // protocol file rename
                        if (isset($_POST['protocol_date'][$row['ID']]) and strcmp($_POST['protocol_date'][$row['ID']], $row['PROC_DATE'])) {
                            $old_file_name = "Протокол_" . $row['PTK'] . "_" . $row['REGION'] . "_" . $row['ENVIRONMENT'] . "_версия_" . $row_ver['PASS_VERSION'] . "_от_" . $row_ver['PROC_DATE'] . "." . $row_ver['PROC_FILE'];
                            $new_file_name = "Протокол_" . $row['PTK'] . "_" . $row['REGION'] . "_" . $row['ENVIRONMENT'] . "_версия_" . $row_ver['PASS_VERSION'] . "_от_" . $_POST['protocol_date'][$row_ver['ID']] . "." . $row_ver['PROC_FILE'];
                            if (file_exists($pass_dir . $old_file_name))
                                if (rename($pass_dir . $old_file_name, $pass_dir . $new_file_name))
                                    $output .= " Файл Протокола \"" . $old_file_name . "\" переименован в \"" . $new_file_name . "\".";
                                else
                                    $output .= " Ошибка переименования Файла Протокола \"" . $old_file_name . "\" в \"" . $new_file_name . "\"!";
                            else
                                $output .= " Файл Протокола \"" . $old_file_name . "\" не найден на диске!";
                        }
                        else
                            $output .= " Отметка об утверждении Протокола установлена.";
                    }
                }

                $output .= " В БД внесены изменения версий Паспорта/Протокола для ".$NODEID;
            }

            // passport upload
            if (isset($_POST['passport_upload'])) {
                // upload passport check
                if (is_uploaded_file($_FILES['passport_file']['tmp_name'])) {
                    // upload file type check
                    $file_type = substr(strrchr($_FILES['passport_file']['name'], '.'), 1);
                    if($file_type == "doc" or $file_type == "docx" or $file_type == "pdf" or $file_type == "zip") {
                        // new file save
                        $file_name = "Паспорт_".$row['PTK']."_".$row['REGION']."_".$row['ENVIRONMENT']."_версия_".$_POST['passport_version']."_от_".$_POST['passport_date'].".".$file_type;
                        if (file_exists($pass_dir.$file_name))
                            $output = "Ошибка! Файл Паспорта с именем \"".$file_name."\" уже существует!";
                        else {
                            if (move_uploaded_file($_FILES['passport_file']['tmp_name'], $pass_dir.$file_name)) {
                                exec("convmv -rf cp1251 -t UTF-8 --notest " . $pass_dir.$file_name);
                                $output .= " Новый файл Паспорта для ".$NODEID." успешно сохранён.";

                                // table insert
                                $sel = "insert into DB2INST1.PFR_PASSPORT_VERSIONS 
                                                                   (FILE_CODE, 
                                                                    PASS_VERSION, 
                                                                    PASS_SIGNED, 
                                                                    PASS_DATE,
                                                                    PASS_FILE) 
                                                            values ('" . $pass_code . "', 
                                                                    '" . $_POST['passport_version'] . "', 
                                                                    '" . (empty($_POST['passport_signed']) ? "N" : "Y") . "', 
                                                                    '" . $_POST['passport_date'] . "',
                                                                    '" . $file_type . "')";
                                $stmt = db2_prepare($connection_TBSM, $sel);
                                $result = db2_execute($stmt);
                                $output .= " Таблицы в БД обновлены.";
                            } else
                                $output .= " Ошибка при сохранении нового файла Паспорта!";
                        }
                    }
                    else {
                        $output .= " Новый файл Паспорта для ".$NODEID." имеет ошибочный тип. Допустимы форматы \"doc\", \"docx\", \"pdf\", \"zip\".";
                        if(unlink($_FILES['passport_file']['tmp_name']))
                            $output .= " Временный файл удалён!";
                        else
                            $output .= " Ошибка при удалении временного файла!";
                    }
                }
                else {
                    if (empty($_FILES['passport_file']['name']))
                        $output .= " Новый файл Паспорта для ".$NODEID." не задан!";
                    else
                        $output .= " Ошибка при загрузке нового файла Паспорта для ".$NODEID." !";
                }
            }

            // protocol upload
            if (isset($_POST['protocol_upload'])) {
                // upload protocol check
                if (is_uploaded_file($_FILES['protocol_file']['tmp_name'])) {
                    // upload file type check
                    // upload file type check
                    $file_type = substr(strrchr($_FILES['protocol_file']['name'], '.'), 1);
                    if ($file_type == "doc" or $file_type == "docx" or $file_type == "pdf" or $file_type == "zip") {
                        // database table record check
                        $sel = "select PASS_FILE from DB2INST1.PFR_PASSPORT_VERSIONS where FILE_CODE = '$pass_code' and PASS_VERSION = '".$_POST['protocol_for_passport']."'";
                        $stmt = db2_prepare($connection_TBSM, $sel);
                        $result = db2_execute($stmt);

                        $pass_check = false;
                        while ($row_check = db2_fetch_assoc($stmt))
                            if ($file_type == "pdf") {
                                if ($row_check['PASS_FILE'] == "pdf")
                                    $pass_check = true;
                            }
                            else {
                                if ($row_check['PASS_FILE'] == "doc" or $row_check['PASS_FILE'] == "docx" or $row_check['PASS_FILE'] == "zip")
                                    $pass_check = true;
                            }
                        if (!$pass_check)
                            $output = "Ошибка! Для Протокола с расширением $file_type не найден соответствующий тип Паспорта!";
                        else {
                            // new file save
                            $file_name = "Протокол_" . $row['PTK'] . "_" . $row['REGION'] . "_" . $row['ENVIRONMENT'] . "_версия_" . $_POST['protocol_for_passport'] . "_от_" . $_POST['protocol_date'] . "." . $file_type;
                            if (file_exists($pass_dir . $file_name))
                                $output = "Ошибка! Файл Протокола с именем \"" . $file_name . "\" уже существует!";
                            else {
                                if (move_uploaded_file($_FILES['protocol_file']['tmp_name'], $pass_dir . $file_name)) {
                                    exec("convmv -rf cp1251 -t UTF-8 --notest " . $pass_dir . $file_name);
                                    $output .= " Новый файл Протокола для " . $NODEID . " успешно сохранён.";

                                    // table update
                                    $sel = "update DB2INST1.PFR_PASSPORT_VERSIONS set PROC_SIGNED = '" . (empty($_POST['protocol_signed']) ? "N" : "Y") . "', PROC_DATE = '" . $_POST['protocol_date'] . "', PROC_FILE = '" . $file_type . "' where FILE_CODE = '$pass_code' and PASS_VERSION = '" . $_POST['protocol_for_passport'] . "' and PASS_FILE ".($file_type == 'pdf' ? '=' : '<>')." 'pdf'";
                                    $stmt = db2_prepare($connection_TBSM, $sel);
                                    $result = db2_execute($stmt);
                                    $output .= " Таблица в БД обновлена.";
                                } else
                                    $output .= " Ошибка при сохранении нового файла Протокола!";
                            }
                        }
                    }
                    else {
                        $output .= " Новый файл Протокола для ".$NODEID." имеет ошибочный тип. Допустимы форматы \"doc\", \"docx\", \"pdf\", \"zip\".";
                        if(unlink($_FILES['protocol_file']['tmp_name']))
                            $output .= " Временный файл удалён!";
                        else
                            $output .= " Ошибка при удалении временного файла!";
                    }
                }
                else {
                    if (empty($_FILES['protocol_file']['name']))
                        $output .= " Новый файл Протокола для ".$NODEID." не задан!";
                    else
                        $output .= " Ошибка при загрузке нового файла Протокола для ".$NODEID." !";
                }
            }
        }
        echo "<img src='images/ok.png' hspace='10'><br>";
    }

    // buttons from edit/add/clone/delete form
    if (isset($_POST['sendRequest'])) {
        if ($_POST['sendRequest'] == 'save') {
            $PFR_LOCATIONS_fields['ID']['VALUE'] = $_POST['ID_hidden'];
            if (record_form("ServiceName={$NODEID}", 'save'))
                $output = "Запись сохранена в PFR_LOCATIONS";
            else
                $output = "Ошибка сохранения записи в PFR_LOCATIONS";
        }
        if ($_POST['sendRequest'] == 'clone') {
            foreach ($PFR_LOCATIONS_fields as $field => &$prop)
                if ($field != 'ID')
                    $prop['VALUE'] = $_POST[$field];
            unset($prop);
            if (record_form("ServiceName={$NODEID}", 'add'))
                exit();
            else
                $output = "Ошибка отображения записи из PFR_LOCATIONS";
        }
        if ($_POST['sendRequest'] == 'delete') {
            $PFR_LOCATIONS_fields['ID']['VALUE'] = $_POST['ID_hidden'];
            if (record_form("ServiceName={$NODEID}", 'delete'))
                $output = "Запись удалена из PFR_LOCATIONS";
            else
                $output = "Ошибка удаления записи из PFR_LOCATIONS";
        }
    }

    // main web form button was pressed
    if (isset($_POST['formId']['sendRequest']) and (!empty($acs_user) or $_POST['formId']['sendRequest'] == 'Фильтр по полю')) {
        $post_command = explode(':', $_POST['formId']['sendRequest'], 2);
        echo "Обработка нажатия кнопки \"".$post_command[0]."\"...";
        ob_flush();
        flush();

        // incident number check
        $incident = true;
        if ($connection_SCCD and ($post_command[0] == TO_WORK_MODE or $post_command[0] == TO_MAINTENANCE_MODE or $post_command[0] == TO_BE_NOT_BLUE or $post_command[0] == TO_BE_BLUE)) {
            $incident = false;
            // ticket from MAXIMO database check
            $sel = "SELECT TICKETID, CLIENTNAME FROM MAXIMO.TICKET WHERE TICKETID='" . $_COOKIE["incident"] . "' AND \"CLASS\"='SR'";
            $stmt_SCCD = db2_prepare($connection_SCCD, $sel);
            $result = db2_execute($stmt_SCCD);
            $row = db2_fetch_assoc($stmt_SCCD);
            if (!empty($row['TICKETID']))
                $incident = true;
            $initiator = !empty($row['CLIENTNAME']) ? $row['CLIENTNAME'] : '';
        }

        if (!$incident)
            $output = "Несуществующий номер заявки!";
        else {
            // pressed button selection
            switch ($post_command[0]) {
                case 'Фильтр по полю':
                    $table_N2_data[0]['ke'] = $_POST['filter_ke'];
                    $output = "Фильтр к таблице ситуаций применён.";
                    $log_flag = false;
                    break;
                case 'Проверить наличие агента BigFix':
                    exec("curl -X GET --insecure --connect-timeout 10 --user monitoring:tivoli \"https://10.101.237.58:52315/api/query?relevance=exists%28names%20of%20bes%20computers%20whose%28name%20of%20it%20as%20lowercase%20as%20trimmed%20string%20=%20%22" . $NODEID . "%22%20as%20trimmed%20string%20as%20lowercase%29%29\"", $arr_BigFix);
                    $output = "Сервер BigFix не отвечает. Данные о наличии агента не могут быть получены!";
                    foreach ($arr_BigFix as $value)
                        if (strstr($value, "</Answer>")) {
                            $is_BigFix_agent = strstr($value, "True") ? "установлен" : "отсутствует";
                            $output = "Агент BigFix " . $is_BigFix_agent . ".";
                        }
                    break;
                case 'Обновить данные для отчётов':
                    // another user actions detect
                    list($flag_busy, $user_busy, $time_busy,) = explode(';', file_get_contents($report_busy_file));
                    if ($flag_busy == '1') {
                        $output = "Обновление данных для отчётов уже запущено пользователем " . $user_busy . " в " . date('H:i', $time_busy) . ". Повторите действие позднее!";
                        $log_flag = false;
                    }
                    else {
                        file_put_contents($report_busy_file, "1;" . $acs_user . ";" . time(), LOCK_EX);
                        TEMS_data_reload_new($_POST['scripts'], true, isset($_POST['sit_exp']));
                        file_put_contents($report_busy_file, "0;" . $acs_user . ";" . time(), LOCK_EX);
                        $output = "Обновление данных для отчётов завершено.";
                        $log_flag = false;
                    }
                    break;
                case 'Удалить сервис':
                    file_put_contents("/StorageDisk/RAD/Add_instances/service_list.txt", "", LOCK_EX);
                    file_put_contents($undeleted, "", LOCK_EX);

                    foreach ($services as $key => $value) {
                        $i = 0;
                        if ($multi_scheme) {
                            // parent services search
                            $sel_parents = "select par.SERVICEINSTANCENAME as PARENT
                                        from TBSMBASE.SERVICEINSTANCE ch, TBSMBASE.SERVICEINSTANCE par, TBSMBASE.SERVICEINSTANCERELATIONSHIP
                                        where ch.SERVICEINSTANCENAME = '$key' and CHILDINSTANCEKEY = ch.SERVICEINSTANCEID and PARENTINSTANCEKEY = par.SERVICEINSTANCEID and par.SERVICEINSTANCENAME <> '$NODEID' and par.SERVICEINSTANCENAME not like '%OPFR'";
                            $stmt_parents = db2_prepare($connection_TBSM, $sel_parents);
                            $result_parents = db2_execute($stmt_parents);

                            while ($row_parents = db2_fetch_assoc($stmt_parents)) {
                                if (!array_key_exists($row_parents['PARENT'], $services)) {
                                    if ($i == 0)
                                        file_put_contents($undeleted, "\n{$key}", FILE_APPEND | LOCK_EX);
                                    $i++;
                                    file_put_contents($undeleted, "|{$row_parents['PARENT']}", FILE_APPEND | LOCK_EX);
                                }
                            }
                        }

                        // delete services without other parents
                        if ($i == 0) {
                            file_put_contents("/StorageDisk/RAD/Add_instances/service_list.txt", "{$key}\n", FILE_APPEND | LOCK_EX);
                            $sel_delete = "delete from DB2INST1.PFR_LOCATIONS where SERVICE_NAME = '$key'";
                            $stmt_delete = db2_prepare($connection_TBSM, $sel_delete);
                            $result_delete = db2_execute($stmt_delete);
                        }
                    }
                    file_put_contents("/StorageDisk/RAD/Add_instances/service_list.txt", $multi_scheme ? "{$NODEID}\n" : "", FILE_APPEND | LOCK_EX);
                    $output = rtrim(shell_exec("/StorageDisk/RAD/Add_instances/Delete_from_list_web.sh service_list.txt"));
                    $output .= " Из таблицы PFR_LOCATIONS удалены соответствующие записи, кроме сервисов, содержащихся в других контейнерах (см. <a href='http://10.103.0.60/pfr_other/{$undeleted}' target='_blank'>файл</a>)";
                    break;
                case 'Удалить':
                    if (isset($_POST['chkbx_del'])) {
                        foreach ($_POST['chkbx_del'] as $key => $value) {
                            $sel_delete = "delete from DB2INST1.PFR_LOCATIONS where ID = '$key'";
                            $stmt_delete = db2_prepare($connection_TBSM, $sel_delete);
                            $result_delete = db2_execute($stmt_delete);
                        }
                        $output = "Отмеченные записи удалены из PFR_LOCATIONS.";
                    }
                    else {
                        $output = "Не было выбрано ни одной строки для удаления из PFR_LOCATIONS!";
                        $log_flag = false;
                    }
                    break;
                case 'Клонировать':
                    if (isset($_POST['chkbx_del'])) {
                        $sel_id = "select * from DB2INST1.PFR_LOCATIONS where ID = ".key($_POST['chkbx_del']);
                        $stmt_id = db2_prepare($connection_TBSM, $sel_id);
                        $result_id = db2_execute($stmt_id);
                        $row_id = db2_fetch_assoc($stmt_id);

                        foreach ($PFR_LOCATIONS_fields as $field => &$prop)
                            $prop['VALUE'] = $row_id[$field];
                        unset($prop);
                        if (record_form("ServiceName={$NODEID}", 'edit'))
                            exit();
                        else {
                            $output = "Ошибка отображения записи из PFR_LOCATIONS";
                            $log_flag = false;
                        }
                    }
                    else {
                        $output = "Не было выбрано ни одной строки для клонирования!";
                        $log_flag = false;
                    }
                    break;
                case 'Сохранить изменения в PFR_LOCATIONS':
                    foreach ($_POST['txtfld_loc'] as $key => $value) {
                        $sel_update = "UPDATE DB2INST1.PFR_LOCATIONS SET NODE = '{$value['node']}', PFR_OBJECT = '{$value['obj']}', PFR_KE_TORS = '{$value['ke']}', SITFILTER = '{$value['filt']}', PFR_NAZN = '{$value['nazn']}', URL = '{$value['url']}' WHERE ID = '$key'";
                        $stmt_update = db2_prepare($connection_TBSM, $sel_update);
                        $result_update = db2_execute($stmt_update);
                    }
                    $output = "Изменения данных в PFR_LOCATIONS сохранены.";
                    break;
                case 'Включить':
                case 'Отключить':
                    if (isset($_POST['chkbx_inc'])) {
                        foreach ($_POST['chkbx_inc'] as $key => $value) {
                            // change send incident status in PFR_LOCATIONS table
                            $sel_update = "UPDATE DB2INST1.PFR_LOCATIONS SET INCIDENT_SEND = ".($post_command[0] == 'Включить' ? 1 : -1)." WHERE ID = '$key'";
                            $stmt_update = db2_prepare($connection_TBSM, $sel_update);
                            $result_update = db2_execute($stmt_update);

                            // select SERVICE_NAME, PFR_KE_TORS, NODE from PFR_LOCATIONS table
                            $sel_data = "select SERVICE_NAME, PFR_KE_TORS, NODE from PFR_LOCATIONS WHERE ID = '$key'";
                            $stmt_data = db2_prepare($connection_TBSM, $sel_data);
                            $result_data = db2_execute($stmt_data);
                            $row_data = db2_fetch_assoc($stmt_data);

                            // add affected nodes to services array
                            $services[$row_data['SERVICE_NAME']]['inc_change'][] = $row_data['NODE'];

                            // alerts.status update
                            $command = "UPDATE alerts.status SET ImpactFlag='UPDATED' WHERE pfr_ke_tors = '{$row_data['PFR_KE_TORS']}'";
                            file_put_contents($sql_file, $command . ";\ngo\n", LOCK_EX);
                            shell_exec("/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password ... -server NCOMS < $sql_file");
                        }
                        $output = "Отправка инцидентов по выбранным строкам ".($post_command[0] == 'Включить' ? 'включена' : 'отключена').".";
                    }
                    else {
                        $output = "Не было выбрано ни одной строки для изменения статусов отправки инцидентов!";
                        $log_flag = false;
                    }
                    break;
                case 'Выполнить действие':
                    $incident_write = $_COOKIE["incident"];
                    switch ($_POST['maint']) {
                        case 'to_maint':
                            file_put_contents("/StorageDisk/RAD/Maintenance/service_list.txt", "{$NODEID}\n");
                            $output = rtrim(shell_exec("/StorageDisk/RAD/Maintenance/manage.sh 0"));
                            if ($output == "Перевод в режим обслуживания завершен.")
                                $maint_status = 'unprepared';
                            break;
                        case 'from_maint':
                            file_put_contents("/StorageDisk/RAD/Maintenance/service_list.txt", "{$NODEID}\n");
                            $output = rtrim(shell_exec("/StorageDisk/RAD/Maintenance/manage.sh 1"));
                            if ($output == "Вывод из режима обслуживания завершен.")
                                $maint_status = '';
                            break;
                        case 'ind_idle':
                        default:
                            break;
                    }
                    switch ($_POST['incid']) {
                        case 'inc_off':
                            $all_inc_send_change = -1;
                        case 'inc_idle':
                            if (isset($_POST['inc_snap']))
                                $snapshot_action = 'save';
                            break;
                        case 'inc_on':
                            $all_inc_send_change = 1;
                            break;
                        case 'inc_restore':
                            $snapshot_action = 'restore';
                            break;
                        default:
                            break;
                    }
                    break;
                case 'Отправить все в очередь':
                    echo "<script>window.open('http://10.103.0.60/pfr_other/SCCD_queue.php?service=" . $NODEID . "&lifetime=" . ($light_scheme ? 0 : $event_lifetime) . "')</script>";
                    break;
                case 'Запустить':
                    $run_event = $post_command[1];
                    break;
                case 'Запустить все':
                    $run_event = 'all';
                    break;
                default:
                    break;
            }
        }

        echo "<img src='images/ok.png' hspace='10'><br>";
    }

    // incident status change and find all KE
    echo "Выбор записей из PFR_LOCATIONS...&emsp;";
    ob_flush();
    flush();

    $snapshot_time = date('d.m.Y H:i:s');
    $save_res = $restore_res = true;
    foreach ($services as $key => $value) {
        // record(s) selection from PFR_LOCATIONS table
        $sel = "SELECT NODE, ID, SUBCATEGORY, (CASE WHEN AGENT_NODE = '' THEN NODE ELSE AGENT_NODE END) AS AGENT_PLACE, PFR_OBJECT, PFR_OBJECTSERVER, PFR_NAZN, PFR_KE_TORS, INCIDENT_SEND, AGENT_NODE, SITFILTER, PFR_ID_TORG, TEMS, URL
                FROM DB2INST1.PFR_LOCATIONS 
                WHERE SERVICE_NAME = '$key'
                ORDER BY SUBCATEGORY, (CASE WHEN AGENT_NODE = '' THEN NODE ELSE AGENT_NODE END) ASC";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);

        while ($row = db2_fetch_assoc($stmt)) {
            // snapshot save
            if ($snapshot_action == 'save') {
                if (empty($snapshot))
                    $sel_save = "insert into DB2INST1.PFR_MAINTENANCE_SNAPSHOTS (SERVICE_NAME, LOC_ID, INCIDENT_SEND, TIMESTAMP) values ('{$NODEID}', {$row['ID']}, {$row['INCIDENT_SEND']} ,'{$snapshot_time}')";
                else
                    $sel_save = "update DB2INST1.PFR_MAINTENANCE_SNAPSHOTS set TIMESTAMP = '{$snapshot_time}', INCIDENT_SEND = {$row['INCIDENT_SEND']} where SERVICE_NAME = '{$NODEID}' and LOC_ID = {$row['ID']}";
                $stmt_save = db2_prepare($connection_TBSM, $sel_save);
                $result_save = db2_execute($stmt_save);
                if ($save_res)
                    $save_res = db2_num_rows($stmt_save) == 1 ? true : false;
            }

            // if all incidents off or on
            if ($all_inc_send_change != 0)
                $row['INCIDENT_SEND'] = $all_inc_send_change;

            // if incidents restore
            if ($snapshot_action == 'restore') {
                $sel_restore = "select INCIDENT_SEND from DB2INST1.PFR_MAINTENANCE_SNAPSHOTS where SERVICE_NAME = '{$NODEID}' and LOC_ID = {$row['ID']}";
                $stmt_restore = db2_prepare($connection_TBSM, $sel_restore);
                $result_restore = db2_execute($stmt_restore);
                $row_restore = db2_fetch_assoc($stmt_restore);
                $row['INCIDENT_SEND'] = empty($row_restore) ? $row['INCIDENT_SEND'] : $row_restore['INCIDENT_SEND'];
                if ($restore_res)
                    $restore_res = empty($row_restore) ? false : true;

            }

            // incident status update in PFR_LOCATIONS table
            $sel_update = "UPDATE DB2INST1.PFR_LOCATIONS SET INCIDENT_SEND = ".$row['INCIDENT_SEND']." WHERE ID = '".$row['ID']."'";
            $stmt_update = db2_prepare($connection_TBSM, $sel_update);
            $result_update = db2_execute($stmt_update);

            // alerts.status update
            $command = "UPDATE alerts.status SET ImpactFlag='UPDATED' WHERE pfr_ke_tors = '" . $row['PFR_KE_TORS']. "'";
            file_put_contents($sql_file, $command . ";\ngo\n", LOCK_EX);
            shell_exec("/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password ... -server NCOMS < $sql_file");

            // incident send status
            if ($row['INCIDENT_SEND'] == 1) {
                $all_inc_send_on = isset($all_inc_send_on) ? $all_inc_send_on : true;
                $all_inc_send_off = false;
            }
            if ($row['INCIDENT_SEND'] == -1) {
                $all_inc_send_on = false;
                $all_inc_send_off = isset($all_inc_send_off) ? $all_inc_send_off : true;
            }

            if (!$light_scheme) {
                $index = array_search($row['PFR_KE_TORS'] . "~" . $key, array_column($pfr_ke_tors, 'unique'));
                // unique combination (ke + service) check
                if ($index === false)
                    $pfr_ke_tors[] = array(
                        'ke' => $row['PFR_KE_TORS'],
                        'service' => $key,
                        'reg' => $row['PFR_ID_TORG'],
                        'unique' => $row['PFR_KE_TORS'] . "~" . $key,
                    );
                // or new region
                else if (!in_array($row['PFR_ID_TORG'], explode(', ', $pfr_ke_tors[$index]['reg'])))
                    $pfr_ke_tors[$index]['reg'] = $pfr_ke_tors[$index]['reg'].', '.$row['PFR_ID_TORG'];

                // array string for PFR_LOCATIONS table
                $table_N1_data[] = array(
                    "ID" => $row['ID'],
                    "SERVICE" => $key,
                    "SUBCATEGORY" => $row['SUBCATEGORY'],
                    "AGENT_PLACE" => $row['AGENT_PLACE'],
                    "PFR_OBJECTSERVER" => $row['PFR_OBJECTSERVER'],
                    "TEMS" => $row['TEMS'],
                    "PFR_ID_TORG" => $row['PFR_ID_TORG'],
                    "EVENT_HISTORY" => "<a href='event_history_new.php?ServiceName=&KE_OBJECT=" . $row['PFR_KE_TORS'] . "&TimeRange=" . date("Y-m-d") . "' target='_blank'><img src='images/events.png' title='Перейти к журналу событий КЭ'></a>",
                    "NODE" => $row['NODE'],
                    "PFR_OBJECT" => $row['PFR_OBJECT'],
                    "PFR_KE_TORS" => $row['PFR_KE_TORS'],
                    "SITFILTER" => $row['SITFILTER'],
                    "PFR_NAZN" => $row['PFR_NAZN'],
                    "URL" => $row['URL'],
                    "INCIDENT_SEND" => $row['INCIDENT_SEND'],
                );
            }
        }

    }
    if ($snapshot_action == 'save') {
        $output .= ($save_res ? ' Резервная копия настроек отправки инцидентов создана.' : 'При создании резервной копии настроек отправки инцидентов произошла ошибка.');
        $snapshot = $save_res ? $snapshot_time : '';
    }
    if ($snapshot_action == 'restore') {
        $output .= ($restore_res ? ' Настройки отправки инцидентов восстановлены из резервной копии.' : 'При восстановлении настроек отправки инцидентов из резервной копии произошла ошибка.');
    }
    if ($all_inc_send_change != 0 and $snapshot_action != 'restore')
        $output .= " Отправка всех инцидентов ".($all_inc_send_change == 1 ? 'включена' : 'отключена').".";

    $count_service_model = count($table_N1_data);
    echo $count_service_model . "<img src='images/ok.png' hspace='10'><br>";

    // find all sources
    if (!$light_scheme) {
        echo "Выбор источников данных...&emsp;";
        ob_flush();
        flush();

        foreach ($pfr_ke_tors as $ke_tors) {
            $sel = "SELECT DISTINCT (CASE WHEN SUBCATEGORY <> 'TSPC_NODE' AND SUBCATEGORY <> 'SYSTEMS_DIRECTOR_NODE' THEN REGION ELSE SUBCATEGORY END) AS CODE FROM DB2INST1.PFR_TEMS_SIT_AGGR WHERE PFR_KE_TORS = '" . $ke_tors['ke'] . "'";
            $stmt = db2_prepare($connection_TBSM, $sel);
            $result = db2_execute($stmt);

            while ($row = db2_fetch_assoc($stmt))
                if (!empty($row['CODE']) and !in_array($row['CODE'], $array_sources))
                    $array_sources[] = $row['CODE'];
        }

        $count_array_sources = count($array_sources);
        echo $count_array_sources . "<img src='images/ok.png' hspace='10'><br>";
    }

    // find formula description for situations
    if (!$light_scheme) {
        echo "Выбор ситуаций и описаний из ТОРС...&emsp;";
        ob_flush();
        flush();

        if ($connection_SCCD) {
            foreach ($pfr_ke_tors as $ke_tors) {
                // record(s) selection from PFR_TEMS_SIT_AGGR table
                $sel = "SELECT SIT_NAME FROM DB2INST1.PFR_TEMS_SIT_AGGR WHERE PFR_KE_TORS = '{$ke_tors['ke']}' AND SERVICE_NAME = '{$ke_tors['service']}' 
                UNION 
                SELECT SIT_NAME FROM DB2INST1.PFR_TEMS_SIT_OVER WHERE PFR_KE_TORS = '{$ke_tors['ke']}' AND SERVICE_NAME = '{$ke_tors['service']}'";
                $stmt = db2_prepare($connection_TBSM, $sel);
                $result = db2_execute($stmt);

                while ($row = db2_fetch_assoc($stmt))
                    $form_descr_arr[] = $row['SIT_NAME'];
            }

            $form_descr_arr_unique = array_unique($form_descr_arr);
            $form_descr_arr = [];

            // get formula description from SCCD
            $sel = "SELECT CODSIT, NAME, FORMULADESC FROM MAXIMO.TEMPCI WHERE NAME in ('" . implode("', '", $form_descr_arr_unique) . "')";
            $stmt_SCCD = db2_prepare($connection_SCCD, $sel);
            $result_SCCD = db2_execute($stmt_SCCD);
            while ($row_SCCD = db2_fetch_assoc($stmt_SCCD))
                $form_descr_arr[$row_SCCD['NAME']] = array(
                    'code' => $row_SCCD['CODSIT'],
                    'formula' => $row_SCCD['FORMULADESC'],
                );

            $count_form_descr = count($form_descr_arr);
            echo $count_form_descr . "<img src='images/ok.png' hspace='10'><br>";
        }
        else {
            $form_descr_arr = [];
            echo "СОЕДИНЕНИЕ С БД СТП ОТСУТСТВУЕТ!<img src='images/error.png' hspace='10'><br>";
        }
    }

    // find all situations
    if (!$light_scheme) {
        echo "Выбор ситуаций из мониторинга и AEL".($run_event == 'all' ? " c генерацией тестовых событий" : "")."...&emsp;";
        ob_flush();
        flush();

        $i = 0;
        foreach ($pfr_ke_tors as $ke_tors) {
            // record(s) selection from PFR_TEMS_SIT_AGGR table
            $sel = "SELECT * FROM DB2INST1.PFR_TEMS_SIT_AGGR WHERE PFR_KE_TORS = '{$ke_tors['ke']}' AND SERVICE_NAME = '{$ke_tors['service']}' 
                    UNION 
                    SELECT * FROM DB2INST1.PFR_TEMS_SIT_OVER WHERE PFR_KE_TORS = '{$ke_tors['ke']}' AND SERVICE_NAME = '{$ke_tors['service']}'";
            $stmt = db2_prepare($connection_TBSM, $sel);
            $result = db2_execute($stmt);

            while ($row = db2_fetch_assoc($stmt)) {
                // find formula description
                $sit_form_descr = array_key_exists($row['SIT_NAME'], $form_descr_arr) ? $form_descr_arr[$row['SIT_NAME']]['formula'] : '';

                // unique combination (place + ke + sit_name + region) check
                $new_place = $row['AGENT_NODE'] == '' ? $row['NODE'] : $row['AGENT_NODE'];
                $new_ke = $row['SIT_NAME'] != "MS_Offline" ? $row['PFR_KE_TORS'] : "---";
                $new_sit_name = $row['SIT_NAME'] . ($row['SIT_NAME'] == "MS_Offline" ? " (агент " . $row['SUBCATEGORY'] . ")" : "");
                $new_region = $row['REGION'];
                $new_unique = $new_place."~".$new_ke."~".$new_sit_name."~".$new_region;

                // array of strings for situations table ('OFFLINE' renames to '!OFFLINE' for the purpose of easy sorting)
                $key = array_search($new_unique, array_column($table_N2_data, 'unique'));
                if ($key === false) {
                    $table_N2_data[] = array(
                        "number" => $i++,
                        "place" => $new_place,
                        "ke" => $new_ke,
                        "url" => array($row['DESCRIPTION']),
                        "hubs" => array($row['STATUS'] => array($row['TEMS'])),
                        "sit_name" => $new_sit_name,
                        "sit_code" => $row['SIT_CODE'] == 'OFFLINE' ? '!OFFLINE' : (empty($row['SIT_CODE']) ? '-' : $row['SIT_CODE']),
                        "descr" => $sit_form_descr,
                        "sit_form" => $row['FORMULA'],
                        "severity" => $row['SEVERITY'],
                        "period" => (!empty($row['INTERVAL']) ? substr($row['INTERVAL'], 0, 2) . ":" . substr($row['INTERVAL'], 2, 2) . ":" . substr($row['INTERVAL'], 4, 2) : "00:00:00") . "<br>" . $row['COUNT'],
                        "unique" => $new_unique,
                    );
                }
                else {
                    // add new URL for the same KE
                    $table_N2_data[$key]['url'][] = $row['DESCRIPTION'];
                    // add new HUB with the situation's status
                    $table_N2_data[$key]['hubs'][$row['STATUS']][] = $row['TEMS'];
                }

                if ($key === false) {
                    // run test event(s)
                    if ($run_event == 'all' or $run_event == $new_unique) {
                        // sql command form
                        $event_id = $row['NODE'] . ':' . $row['SIT_CODE'] . ':' . (isset($_POST['const_ID']) ? '777' : rand(100, 999));
                        $reg_code = $row['REGION'] == '092' ? '091' : $row['REGION'];
                        $NCO = $reg_code == '101' ? 'NCOMS' : 'NCO' . $reg_code;
                        $command = "INSERT INTO alerts.status (Identifier, Severity, Class, Manager, FirstOccurrence, LastOccurrence, ExpireTime, Node, NodeAlias, Summary, pfr_description, ITMApplLabel, pfr_sit_name, ServerName, ITMDisplayItem) VALUES ('" . $event_id . "', " . $severity_codes[$row['SEVERITY']] . ", " . (array_key_exists($row['SIT_NAME'], $event_codes) ? $event_codes[$row['SIT_NAME']] : '87722') . ", 'tivoli_eif probe test event', getdate(), getdate(), " . $event_lifetime . ", '" . $row['NODE'] . "', '" . $row['NODE'] . "', 'ТЕСТ РЗ: " . str_replace(array("\"", "'"), "", $sit_form_descr) . "', 'ТЕСТ РЗ: " . str_replace(array("\"", "'"), "", $sit_form_descr) . "', '" . $row['CATEGORY'] . "', '" . $row['SIT_CODE'] . "', '" . $NCO . "', 'ТЕСТ РЗ')";

                        // sql command write to file and execute
                        file_put_contents($sql_file, $command . ";\ngo\n", LOCK_EX);
                        usleep(500000);
                        $q_string = shell_exec("/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password ... -server NCOMS < $sql_file");
                        $q_status = empty(strcmp(trim($q_string), "(1 row affected)"));
                        if (!$q_status)
                            $err_num++;
                        file_put_contents($log_file_sql, date('d.m.Y H:i:s') . "\t" . ($q_status ? "Y" : "N") . "\t" . $command . ";\n", FILE_APPEND | LOCK_EX);

                        // log file and PFR_TEST_EVENTS_LOG table write
                        $output = "Тестовое событие для ситуации " . $row['SIT_NAME'] . ($q_status ? "" : " не ") . " сгенерировано" . ($q_status ? " (" . $event_id . ")." : "!");
                        file_put_contents($log_file, date('d.m.Y H:i:s')."\t".$NODEID." (".$NODEDESCRIPTION.")\t".$acs_user."\t".$output."\t".$_COOKIE["comment"]."\t\n", FILE_APPEND | LOCK_EX);
                        $sel_evnt = "insert into DB2INST1.PFR_TEST_EVENTS_LOG (SERVICE_NAME, DISPLAY_NAME, USER, OPERATION, DESCRIPTION, TIMESTAMP, INCIDENT, INITIATOR) 
                                     values ('".$NODEID."', '".$NODEDESCRIPTION."', '".$acs_user."', '".$output."', '".$_COOKIE["comment"]."', CURRENT TIMESTAMP)";
                        $stmt_evnt = db2_prepare($connection_TBSM, $sel_evnt);
                        $result_evnt = db2_execute($stmt_evnt);
                    }
                }
            }
            // AEL
            foreach (ael_request('pfr_ke_tors', $ke_tors['ke']) as $key => $val)
                if (!array_key_exists($key, $table_N1_1_data))
                    $table_N1_1_data[$key] = $val;
        }
        // AEL array sort
        usort ($table_N1_1_data, function ($x, $y) {
            return ($x['Severity'] < $y['Severity']);
        });

        // AEL incidents close
        if ($all_inc_send_change == -1 and isset($_POST['inc_close'])) {
            $i = 0;
            $inc_numbers_arr = [];
            foreach($table_N1_1_data as $value) {
                if (!empty($value['TTNumber'])) {
                    $inc_number = $value['TTNumber'];
                    $arr_AELoff = [];
                    exec("curl -X POST -d '<UpdateINCIDENT xmlns=\"http://www.ibm.com/maximo\" creationDateTime=\"" . date('c') . "\" transLanguage=\"EN\" messageID=\"123\" maximoVersion=\"7.5\"> <INCIDENTSet><INCIDENT action=\"Change\"><STATUS><![CDATA[CLOSED]]></STATUS><TICKETID>{$inc_number}</TICKETID><WORKLOG action=\"Add\"><CREATEDATE>" . date('c') . "</CREATEDATE><DESCRIPTION_LONGDESCRIPTION><![CDATA[Инцидент закрыт в связи с тех. работами на КЭ]]></DESCRIPTION_LONGDESCRIPTION></WORKLOG></INCIDENT></INCIDENTSet></UpdateINCIDENT>' http://tivoli:12345678@{$SCCD_server}/meaweb/es/ITM/INCIDENTUpdate", $arr_AELoff);
                    if (strpos($arr_AELoff[0], 'record does not exist in the database') !== false)
                        $inc_numbers_arr['Инциденты не существуют в БД'][] = $inc_number;
                    else if (strpos($arr_AELoff[0], 'is in history and must remain unchanged') !== false)
                        $inc_numbers_arr['Инциденты уже заархивированы'][] = $inc_number;
                    else if (strpos($arr_AELoff[0], $inc_number) !== false)
                        $inc_numbers_arr['Инциденты закрыты'][] = $inc_number;
                    else
                        $inc_numbers_arr['Ошибки при закрытии инцидентов'][] = "{$inc_number} ({$arr_AELoff[0]})";
                    $i++;
                }
            }
            foreach ($inc_numbers_arr as $key => $inc)
                $output .= " ".$key.": ".implode(', ', $inc);
            if ($i == 0)
                $output .= " Открытых инцидентов не найдено.";
        }

        if ($run_event == 'all')
            $output = "Тестовые события со временем жизни " . array_search($event_lifetime, $arr_event_lifetime) . " для всех ситуаций сгенерированы. Количество ошибок: ".$err_num." Логи см. в " . $log_file_sql;

        $count_table_cells = count($table_N2_data) - 1;
        echo $count_table_cells . "<img src='images/ok.png' hspace='10'><br>";

        // severity search for overridden situations
        $temp_table = &$table_N2_data;
        $temp_severity = '';
        foreach ($table_N2_data as &$row) {
            if ($row['sit_code'] == '-') {
                $needle = $row['sit_name'];
                foreach ($temp_table as &$temp_row) {
                    if ($temp_row['sit_name'] == $needle and $temp_row['sit_code'] != '-') {
                        $temp_severity = $temp_row['severity'];
                        break;
                    }
                }
                unset($temp_row);
                $row['severity'] = $temp_severity;
                $row['test_event'] = str_replace(';-;', ';'.$severity_codes[$temp_severity].';', $row['test_event']);
            }
        }
        unset($row);

        // array sort
        foreach ($table_N2_data as $key => $row) {
            $col_place[$key] = $row['place'];
            $col_ke[$key] = $row['ke'];
            $col_sit_name[$key] = $row['sit_name'];
        }
        array_multisort($col_place, SORT_ASC, $col_ke, SORT_ASC, $col_sit_name, SORT_ASC, $table_N2_data);

        // '!OFFLINE' renames back to 'OFFLINE'
        foreach ($table_N2_data as &$row) {
            if ($row['sit_code'] == '!OFFLINE')
                $row['sit_code'] = 'OFFLINE';
        }
        unset($row);

        // situations templates for constructor
        $sel = "SELECT DISTINCT PFR_SIT_NAME FROM DB2INST1.PFR_SITS_CONSTRUCTOR";
        $stmt = db2_prepare($connection_TBSM, $sel);
        $result = db2_execute($stmt);
        while ($row = db2_fetch_assoc($stmt))
            $sit_templ_arr[] = $row['PFR_SIT_NAME'];
    }

    // fill array with MAXIMO data
    if (!$light_scheme) {
        echo "Выбор ситуаций из ТОРС...&emsp;";
        ob_flush();
        flush();

        if ($connection_SCCD) {
            foreach (array_filter($pfr_ke_tors) as $ke_tors) {
                $sel = "SELECT ci.CINAME, ci.STATUS, ci.ASSETLOCSITEID, cl.CLASSSTRUCTUREID, cl.FAILURECODE, cl.INCTYPEDESC, cl.DELAYMIN, str.CLASSIFICATIONID, str.DESCRIPTION, str.PERSONGROUP, p.DISPLAYNAME
                FROM 
                    (SELECT CINUM, CINAME, STATUS, ASSETLOCSITEID FROM MAXIMO.CI WHERE CINAME = '" . $ke_tors['ke'] . "') AS ci
                LEFT JOIN MAXIMO.CICLASS cl
                ON ci.CINUM = cl.CINUM
                    LEFT JOIN MAXIMO.CLASSSTRUCTURE str
                    ON cl.CLASSSTRUCTUREID = str.CLASSSTRUCTUREID
                        LEFT JOIN MAXIMO.PERSONGROUPTEAM pgt 
                        ON str.PERSONGROUP = pgt.PERSONGROUP AND pgt.SITEDEFAULT = 1 AND pgt.TEAMLEAD = 1 AND pgt.USEFORSITE = ci.ASSETLOCSITEID
                            LEFT JOIN MAXIMO.PERSON p 
                            ON pgt.RESPPARTYGROUP = p.PERSONID
                    ORDER BY CINAME, FAILURECODE ASC";
                $stmt_SCCD = db2_prepare($connection_SCCD, $sel);
                $result = db2_execute($stmt_SCCD);

                while ($row = db2_fetch_assoc($stmt_SCCD)) {
                    $table_N3_data[] = array(
                        "CINAME" => $row['CINAME'],
                        "STATUS" => $row['STATUS'],
                        "CLASSIFICATIONID" => $row['CLASSIFICATIONID'],
                        "CLASSSTRUCTUREID" => $row['CLASSSTRUCTUREID'],
                        "FAILURECODE" => $row['FAILURECODE'],
                        "SITNAME" => '',
                        "INCTYPEDESC" => $row['INCTYPEDESC'],
                        "DESCRIPTION" => $row['DESCRIPTION'],
                        "DELAYMIN" => $row['DELAYMIN'],
                        "PERSONGROUP" => $row['PERSONGROUP'],
                        "DISPLAYNAME" => $row['DISPLAYNAME'],
                        "MON_REG" => $ke_tors['reg'],
                        "ASSETLOCSITEID" => $row['ASSETLOCSITEID'],
                    );
                    $max_delay = $max_delay > $row['DELAYMIN'] ? $max_delay : $row['DELAYMIN'];
                }
            }
            $count_tors_info = count($table_N3_data);
            echo $count_tors_info . "<img src='images/ok.png' hspace='10'><br>";
        }
        else {
            $count_tors_info = 0;
            echo "СОЕДИНЕНИЕ С БД СТП ОТСУТСТВУЕТ!<img src='images/error.png' hspace='10'><br>";
        }
    }

    if (!$light_scheme) {
        echo "Подготовка модели DOM...&emsp;";
        ob_flush();
        flush();
        echo round(((memory_get_usage() - $startMemory)/1024/1024))." МБ<img src='images/ok.png' hspace='10'><br><br>";
        ob_flush();
        flush();
    }
    echo "</p>";
    ?><script>$("#to_remove").remove();</script><?php

    // top header
    $parameters = "?ServiceName={$NODEID}";
    $title = "Настройка интеграции с СТП";
    $links = array (
        "<a href=\"Documents/Инструкция по переводу объекта мониторинга в режим обслуживания.docx\">Инструкция по переводу в режим обслуживания</a>",
        "<a href=\"Documents/Инструкция по генерации тестового события в интерфейсе руководителя TIP.docx\">Инструкция по генерации тестового события</a>",
        "&nbsp;",
        "<a href='SCCD_log_view.php?container={$multi_scheme}&service={$NODEID}' target='_blank'>Просмотреть журнал действий</a>",
        "<a href='SCCD_test_events_view.php?container={$multi_scheme}&service={$NODEID}' target='_blank'>Просмотреть журнал отправки тестовых событий</a>",
        "<a href=\"SCCD_users_manage.php\" target=\"_blank\">Добавить нового пользователя</a>",
    );

    // top header
    $parameters = isset($parameters) ? $parameters : '';
    echo "<table width='100%' border='0' cellspacing='0' cellpadding='10' class='page_title'>";
        echo "<tr>";
            echo "<td width='20%' align='left' rowspan='0' class='page_title_dark'>";
                echo "<div id='code_title_area'>";
                    echo isset($_COOKIE["username"]) ? "{$acs_user} ({$roles_arr[$acs_role]})<br><br>" : "Для активации кнопок введите код доступа:<br><br>";
                echo "</div>";
                echo "<div id='code_reset_area' ".(isset($_COOKIE["username"]) ? '' : "hidden='hidden'").">";
                    echo "<form id='authId_reset'>
                           <input id='code_reset' type='password' value='reset' hidden>
                           <input id='authIdBtn_reset' type='button' class='btn' value='Сбросить код доступа' title='Вернуться к анонимному доступу'>
                          </form>";
                echo "</div>";
                echo "<div id='code_input_area' ".(isset($_COOKIE["username"]) ? "hidden='hidden'" : '').">";
                    echo "<form id='authId_input'>
                            <input id='code_input' type='password' size='30' maxlength='32' required autofocus='true'>
                            <input id='authIdBtn_input' type='button' class='btn' value='OK' title='Перейти к авторизованному доступу'>
                          </form>";
                echo "</div>";
            echo "</td>";
            echo "<td align='center'>";
            echo "<h3>{$title}</h3>";
            echo "</td>";
            echo "<td width='25%' align='right' rowspan='0'>";
            foreach ($links as $link)
                echo "$link<br>";
            echo "</td>";
        echo "</tr>";
        echo "<tr>";
            echo "<td align='center'>";
            echo "<p id='to_remove'><b><font color='red'>Пожалуйста, подождите...   </font></b><img src='images/inprogress.gif' hspace=10></p>";
            // output buffer flush
            ob_flush();
            flush();
            ob_end_flush();

            // top header informational message output
            $db_connect_time_str = (($time_TBSM >= 100 or $time_SCCD >= 100) ? ("Медленное подключение к БД! ".($time_TBSM >= 100 ? "TBSM: {$time_TBSM} мс" : '').
                                                                                                               ($time_SCCD >= 100 ? "SCCD: {$time_SCCD} мс" : '')) : '');
            echo "<b><font color='red'><div id='output_area'>{$db_connect_time_str}{$output}</div></font></b>";
            // log file and PFR_ACTIONS_LOG table write
            if ($log_flag and !empty($output) and $acs_user != 'Говорухин Д.В.') {
                $comment = isset($_COOKIE["comment"]) ? $_COOKIE["comment"] : '';
                if (empty($run_event)) {
                    // incident send status changes
                    if (strpos($output, 'инцидентов ') !== false) {
                        foreach ($services as $key => $value) {
                            if (!empty($value['inc_change']))
                                $output .= " Узлы: ".implode(', ', $value['inc_change']);
                            $parent_action = $multi_scheme ? "Для подсистемы {$NODEID}: " : "";
                            file_put_contents($log_file, date('d.m.Y H:i:s') . "\t" . $key . "\t" . $acs_user . "\t" . "{$parent_action}{$output}\t" . $comment . "\t" . $incident_write . "\n", FILE_APPEND | LOCK_EX);
                            $sel = "INSERT INTO DB2INST1.PFR_ACTIONS_LOG (SERVICE_NAME, DISPLAY_NAME, USER, OPERATION, DESCRIPTION, TIMESTAMP, INCIDENT, INITIATOR) 
                                    VALUES ('" . $key . "', '', '" . $acs_user . "', '{$parent_action}{$output}', '" . $comment . "', CURRENT TIMESTAMP, '" . $incident_write . "', '" . $initiator . "')";
                            $stmt = db2_prepare($connection_TBSM, $sel);
                            $result = db2_execute($stmt);
                        }
                    }
                    if (strpos($output, 'инцидентов ') === false or $multi_scheme) {
                        file_put_contents($log_file, date('d.m.Y H:i:s') . "\t" . $NODEID . " (" . $NODEDESCRIPTION . ")\t" . $acs_user . "\t" . $output . "\t" . $comment . "\t" . $incident_write . "\n", FILE_APPEND | LOCK_EX);
                        $sel = "INSERT INTO DB2INST1.PFR_ACTIONS_LOG (SERVICE_NAME, DISPLAY_NAME, USER, OPERATION, DESCRIPTION, TIMESTAMP, INCIDENT, INITIATOR) 
                                    VALUES ('" . $NODEID . "', '" . $NODEDESCRIPTION . "', '" . $acs_user . "', '" . $output . "', '" . $comment . "', CURRENT TIMESTAMP, '" . $incident_write . "', '" . $initiator . "')";
                        $stmt = db2_prepare($connection_TBSM, $sel);
                        $result = db2_execute($stmt);
                    }
                }
            }
            echo "</td>";
        echo "</tr>";
        echo "<tr>";
            echo "<td align=\"center\">";
            echo "</td>";
        echo "</tr>";
    echo "</table>";

    // *********************************************************** PASSPORT UPLOAD ***********************************************************

    // SERVICESLANAME verify
    if ($template == 'PFR_SERVICE:Standard') {
        // passport record find in PFR_PASSPORT_* tables
        $sel_pass = "select * from DB2INST1.PFR_PASSPORT_CODING where FILE_CODE = 
                  (select FILE_CODE from DB2INST1.PFR_PASSPORT_SERVICES where SERVICE_NAME = '$NODEID')";
        $stmt_pass = db2_prepare($connection_TBSM, $sel_pass);
        $result_pass = db2_execute($stmt_pass);
        $row_pass = db2_fetch_assoc($stmt_pass);

        echo "<br><h3>Сведения о Паспортах мониторинга и Протоколах</h3>";
        if (empty($row_pass))
            echo "Данные не найдены. Для загрузки Паспортов и Протоколов по данной подсистеме необходимо выполнить предварительную настройку на стороне Мониторинга...";
        else {
            // passports info output
            echo "<table border=1 cellpadding=10>";
                echo "<tr>";
                    echo "<td colspan='0' align='center'>";
            echo "Имя сервиса: <b>" . $NODEID . "</b>";
            echo "&emsp;&emsp;&emsp;Отображаемое имя: <b>" . $NODEDESCRIPTION . "</b>";
            echo "&emsp;&emsp;&emsp;Среда: <b>" . $row_pass['ENVIRONMENT'] . "</b>";
            echo "</td>";
            echo "</tr>";
            echo "<tr>";
            echo "<td>";
            echo "Документ:";
            echo "</td>";
            echo "<th>";
            echo "Паспорт";
            echo "</th>";
            echo "<th>";
            echo "Протокол";
            echo "</th>";
            echo "</tr>";
            echo "<tr>";
            echo "<td>";
            echo "Наименование в <a href='http://10.103.0.60/pfr_other/passports.php' target='_blank'>Общем списке</a>:";
            echo "</td>";
            echo "<td>";
            ?>
            <form action="<?php echo $_SERVER['PHP_SELF']; ?>?ServiceName=<?php echo $NODEID; ?>" method="post"
                  id="passport_rename" onsubmit="return checkCookie()">
                <input class="btn_form" name="passport_name" type="text" size="70" maxlength="256"
                       value="<?php echo $row_pass['PASS_DISPLAY_NAME']; ?>" title="Наименование Паспорта в Общем списке"
                       required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />&emsp;
                <input class="btn_form" type="submit" class="btn_blue" name="passport_rename"
                       value="Изменить" title='Изменить наименование Паспорта в Общем списке' <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />
            </form> <?php
            echo "</td>";
            echo "<td>";
            ?>
            <form action="<?php echo $_SERVER['PHP_SELF']; ?>?ServiceName=<?php echo $NODEID; ?>" method="post"
                  id="protocol_rename" onsubmit="return checkCookie()">
                <input class="btn_form" name="protocol_name" type="text" size="70" maxlength="256"
                       value="<?php echo $row_pass['PROC_DISPLAY_NAME']; ?>" title="Наименование Протокола в Общем списке"
                       required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />&emsp;
                <input class="btn_form" type="submit" class="btn_blue" name="protocol_rename"
                       value="Изменить" title='Изменить наименование Протокола в Общем списке' <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />
            </form> <?php
            echo "</td>";
            echo "</tr>";
            echo "<tr>";
            echo "<td>";
            echo "Версии:";
            echo "</td>";
            echo "<td colspan='2'>";
            // passport versions records find in PFR_PASSPORT_* tables
            $sel_ver = "SELECT * FROM DB2INST1.PFR_PASSPORT_VERSIONS WHERE FILE_CODE = '" . $row_pass['FILE_CODE'] . "' ORDER BY PASS_VERSION ASC";
            $stmt_ver = db2_prepare($connection_TBSM, $sel_ver);
            $result_ver = db2_execute($stmt_ver);

            // same version groups
            $ver = '';
            $bg_color = "Lavender";
            ?>
            <form action="<?php echo $_SERVER['PHP_SELF']; ?>?ServiceName=<?php echo $NODEID; ?>" method="post"
                  id="versions_edit" onsubmit="return checkCookie()"> <?php
                echo "<table border='1' cellspacing='0' cellpadding='5' align='center'>";
                echo "<tr>";
                echo "<th colspan='4'>Паспорт</th>";
                echo "<th colspan='3'>Протокол</th>";
                echo "</tr>";
                echo "<tr>";
                echo "<td align='center'>Версия</td>";
                echo "<td align='center'>Дата</td>";
                echo "<td align='center'>Утверждён</td>";
                echo "<td align='center'>Файл</td>";
                echo "<td align='center'>Дата</td>";
                echo "<td align='center'>Утверждён</td>";
                echo "<td align='center'>Файл</td>";
                echo "</tr>";
                while ($row_ver = db2_fetch_assoc($stmt_ver)) {
                    // new version group color
                    if ($ver != $row_ver['PASS_VERSION']) {
                        $ver = $row_ver['PASS_VERSION'];
                        $bg_color = ($bg_color == "Lavender" ? "WhiteSmoke" : "Lavender");
                    }

                    echo "<tr>";
                        echo "<td align='center' bgcolor='".$bg_color."'>";
                        if ($acs_form) {
                            ?> <input name="passport_version[<?php echo $row_ver['ID']; ?>]" type="text" size="5"
                                      maxlength="10" title="Номер версии"
                                      value="<?php echo $row_ver['PASS_VERSION']; ?>" required> <?php ;
                        } else
                            echo $row_ver['PASS_VERSION'];
                        echo "</td>";
                        echo "<td  bgcolor='".$bg_color."'>";
                        if ($acs_form) {
                            ?> <input name="passport_date[<?php echo $row_ver['ID']; ?>]" type="date"
                                      title="Дата выпуска версии" value="<?php echo $row_ver['PASS_DATE']; ?>" size="10"
                                      required> <?php ;
                        } else
                            echo $row_ver['PASS_DATE'];
                        echo "</td>";
                        echo "<td align='center' bgcolor='".$bg_color."'>";
                        if ($row_ver['PASS_SIGNED'] <> 'N')
                            echo "<img src='images/ok.png' title='Утверждён'>";
                        else if ($acs_form) {
                            ?> <input name="passport_signed[<?php echo $row_ver['ID']; ?>]" type="checkbox"
                                      title="Утверждён или нет"/> <?php ;
                        }
                        echo "</td>";
                        echo "<td bgcolor='".$bg_color."'>";
                            if (!empty($row_ver['PASS_DATE'])) {
                                $pass_file_name = "Паспорт_" . $row_pass['PTK'] . "_" . $row_pass['REGION'] . "_" . $row_pass['ENVIRONMENT'] . "_версия_" . $row_ver['PASS_VERSION'] . "_от_" . $row_ver['PASS_DATE'] . "." . $row_ver['PASS_FILE'];
                                echo "<img src='images/".$row_ver['PASS_FILE'].".png' height='16' width='16' hspace='8'>";
                                if (file_exists($pass_dir . $pass_file_name))
                                    echo "<a href=\"http://10.103.0.60" . $pass_URL . $pass_file_name . "\" title=\"Открыть Паспорт\">" . $pass_file_name . "</a>";
                                else
                                    echo $pass_file_name;
                            }
                        echo "</td>";
                        echo "<td bgcolor='".$bg_color."'>";
                        if ($acs_form and !empty($row_ver['PROC_DATE'])) {
                            ?> <input name="protocol_date[<?php echo $row_ver['ID']; ?>]" type="date"
                                      title="Дата выпуска версии" value="<?php echo $row_ver['PROC_DATE']; ?>" size="10"
                                      required> <?php ;
                        } else
                            echo $row_ver['PROC_DATE'];
                        echo "</td>";
                        echo "<td align='center' bgcolor='".$bg_color."'>";
                        if ($row_ver['PROC_SIGNED'] <> 'N')
                            echo "<img src='images/ok.png' title='Утверждён'>";
                        else if ($acs_form and !empty($row_ver['PROC_DATE'])) {
                            ?> <input name="protocol_signed[<?php echo $row_ver['ID']; ?>]" type="checkbox"
                                      title="Утверждён или нет"/> <?php ;
                        }
                        echo "</td>";
                        echo "<td bgcolor='".$bg_color."'>";
                            if (!empty($row_ver['PROC_DATE'])) {
                                $proc_file_name = "Протокол_" . $row_pass['PTK'] . "_" . $row_pass['REGION'] . "_" . $row_pass['ENVIRONMENT'] . "_версия_" . $row_ver['PASS_VERSION'] . "_от_" . $row_ver['PROC_DATE'] . "." . $row_ver['PROC_FILE'];
                                echo "<img src='images/".$row_ver['PROC_FILE'].".png' height='16' width='16' hspace='8'>";
                                if (file_exists($pass_dir . $proc_file_name))
                                    echo "<a href=\"http://10.103.0.60" . $pass_URL . $proc_file_name . "\" title=\"Открыть Протокол\">" . $proc_file_name . "</a>";
                                else
                                    echo $proc_file_name;
                            }
                        echo "</td>";
                    echo "</tr>";

                    $last_ver = $row_ver['PASS_VERSION'];
                    $pass_versions[$last_ver] = $pass_file_name;
                }

                if ($acs_form) {
                    echo "<tr>";
                    echo "<td align='center' colspan='0'>";
                    ?> <input type="submit" class="btn_blue" name="data_edit" value="Применить изменения" title='Сохранить изменения, сделанные в разделе версий'/> <?php
                    echo "</td>";
                    echo "</tr>";
                }
                echo "</table>";
                ?> </form> <?php
            echo "</td>";
            echo "</tr>";
            echo "<tr>";
            echo "<tr>";
            echo "<td>";
            echo "Загрузка:";
            echo "</td>";
            echo "<td valign='top'>";
            ?>
            <form enctype="multipart/form-data"
                  action="<?php echo $_SERVER['PHP_SELF']; ?>?ServiceName=<?php echo $NODEID; ?>" method="post"
                  id="passport_upload" onsubmit="return checkCookie()">
                Паспорт: <input class="btn_form" name="passport_file" type="file" size="70" title="Выбор файла Паспорта для загрузки"
                                required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> /><br><br>
                Номер версии: <input class="btn_form" name="passport_version" type="text" size="5" maxlength="10"
                                     title="Номер очередной версии" value="<?php echo sprintf("%03d", $last_ver + 1); ?>"
                                     required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />&emsp;
                Дата: <input class="btn_form" name="passport_date" type="date" title="Дата выпуска версии"
                             value="<?php echo date("d.m.Y"); ?>" size="10"
                             required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />&emsp;
                Утверждён: <input class="btn_form" name="passport_signed" type="checkbox"
                                  title="Утверждён или нет" <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />&emsp;&emsp;&emsp;
                <p align='center'><input class="btn_form" type="submit" class="btn_blue" name="passport_upload"
                                         value="Загрузить" title='Загрузить новую версию Паспорта' <?php echo $acs_form ? '' : "disabled='disabled'"; ?> /></p>
            </form> <?php
            echo "</td>";
            echo "<td valign='top'>";
            ?>
            <form enctype="multipart/form-data"
                  action="<?php echo $_SERVER['PHP_SELF']; ?>?ServiceName=<?php echo $NODEID; ?>" method="post"
                  id="protocol_upload" onsubmit="return checkCookie()">
                Протокол: <input class="btn_form" name="protocol_file" type="file" size="70"
                                 title="Выбор файла Паспорта для загрузки"
                                 required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> /><br><br>
                Дата: <input class="btn_form" name="protocol_date" type="date" title="Дата выпуска версии"
                             value="<?php echo date("d.m.Y"); ?>" size="10"
                             required <?php echo $acs_form ? '' : "disabled='disabled'"; ?> />&emsp;
                Утверждён: <input class="btn_form" name="protocol_signed" type="checkbox"
                                  title="Утверждён или нет" <?php echo $acs_form ? '' : "disabled='disabled'"; ?> /><br><br>
                к Паспорту: <select size="1" name="protocol_for_passport" required>
                    <option value="">выберите версию Паспорта...</option>
                    <?php
                    foreach ($pass_versions as $v => $f) {
                        ?>
                        <option value= <?php echo $v; ?>><?php echo $f; ?></option><?php ;
                    }
                    ?>
                </select><br>
                <p align='center'><input class="btn_form" type="submit" class="btn_blue" name="protocol_upload"
                                         value="Загрузить" title='Загрузить новую версию Протокола' <?php echo $acs_form ? '' : "disabled='disabled'"; ?> /></p>
            </form> <?php
            echo "</td>";
            echo "</tr>";
            echo "</table>";
        }
        echo "<br \><br \><hr>";
    }

    // *********************************************************** PFR_LOCATIONS ***********************************************************
    ?> <form action="<?php echo $_SERVER['PHP_SELF'];?>?ServiceName=<?php echo $NODEID;?>" method="post" id="formId"> <?php

    // web page output
    echo "<table border=\"0\" cellspacing=\"5\" cellpadding=\"0\" width=\"100%\">";
        echo "<tr>";
            echo "<td colspan=2>";
                echo "<br><h3>Настройки ресурсно-сервисной модели мониторинга</h3>";
            echo "</td>";
            echo "<td align=\"right\">";
                if ($acs_role == 'admin' and !$multi_scheme) {
                    ?><input type="submit" class="<?php echo empty($is_BigFix_agent) ? 'btn_blue' : ($is_BigFix_agent == 'установлен' ? 'btn_green' : 'btn_red'); ?>" name="formId[sendRequest]" value="Проверить наличие агента BigFix" title='Проверить наличие агента BigFix на этом сервисе' <?php echo empty($is_BigFix_agent) ? '' : 'disabled'; ?> /><?php
                }
            echo "</td>";
        echo "</tr>";
        echo "<tr>";
            echo "<td width=\"15%\">";
                echo "Источник данных: ";
            echo "</td>";
            echo "<td>&nbsp;&nbsp;&nbsp;";
                echo "БД ресурсно-сервисной модели (TBSM)";
            echo "</td>";
            echo "<td align='right' rowspan='0'>";
                echo "<div class='update_data' ".($acs_role == 'admin' ? "" : "hidden='hidden'>").">";
                    echo "<table>";
                        echo "<tr>";
                            echo "<td align='center'>";
                                echo "<br><input type='submit' class='btn' name='formId[sendRequest]' value='Обновить данные для отчётов' onclick='return dialog_regions()'><br><br>";
                                echo "<label><input type='checkbox' id='reg_chk' name='sit_exp'>&nbsp;с выгрузкой ситуаций из TEMS</label>&emsp;<br><br>";
                            echo "</td>";
                            echo "<td>";
                                echo "<select size='5' id='reg_list' name='scripts[]' multiple>";
                                    foreach ($array_regions as $k => $r)
                                        if ($k != '000' and $k != '092' and $k != '057' and $k != '060' and $k != '088' and $k != '201') {
                                            ?><option value = <?php echo $k; ?>><?php echo $r; ?></option><?php ;
                                        }
                                echo "</select>";
                            echo "</td>";
                        echo "</tr>";
                        echo "<tr>";
                            echo "<td colspan='0'>";
                                // problems with regional situations upload
                                if (!remote_exec('teps-main', 22, 'root', '...', "/StorageDisk/SITUATIONS/check_regions.sh", $log_regions, false))
                                    echo "!!!";
                                $i = 0;
                                foreach (file($log_regions) as $line) {
                                    if ($i == 0) {
                                        echo "<font color=\"red\">Проблемы с выгрузкой из регионов:&emsp;";
                                        list($r, $e) = explode(':', $line) + array(NULL, NULL);
                                        echo "<a href=\"\" title=\"".$e."\">".$r."</a></font>";
                                        $i++;
                                    }
                                    else {
                                        list($r, $e) = explode(':', $line) + array(NULL, NULL);
                                        echo "<font color=\"red\">, <a href=\"\" title=\"".$e."\">".$r."</a></font>";
                                    }
                                }
                            echo "</td>";
                        echo "</tr>";
                    echo "</table>";
                echo "</div>";
            echo "</td>";
        echo "</tr>";
        echo "<tr>";
            echo "<td>";
                echo "Код ".($multi_scheme ? "подсистемы" : "индикатора").": ";
            echo "</td>";
            echo "<td>";
                echo "&nbsp;&nbsp;&nbsp;".$NODEID;
                // availability check
                echo $multi_scheme ? "" : "&emsp;<a href=\"remote_SSH.php?ServiceName=".$NODEID."\" target=\"_blank\"><img src='images/objectview.gif' title='Проверить доступность'></a>";
                // search in the tree
                echo "&emsp;<a href=\"service_search.php?quick={$NODEID}\" target=\"_blank\"><img src='images/tree_search.jpeg' title='Поиск в сервисном дереве'></a>";
                // remove from service tree
                echo "&emsp;<button class='update_data' type='submit' name='formId[sendRequest]' onclick=\"return (confirm('Будет произведено удаление ".($count_services > 1 ? "{$count_services} сервисов" : "сервиса")." из TBSM и соответствующих записей из PFR_LOCATIONS. Вы уверены?..') && checkCookie())\" value='Удалить сервис' title='Удалить ".($multi_scheme ? "подсистему" : "индикатор")." из сервисного дерева TBSM и записи из PFR_LOCATIONS' ".($acs_role == 'admin' ? "" : "hidden='hidden'")."><img src='images/delete.png'></button>";
        echo "</td>";
        echo "</tr>";
        echo "<tr>";
            echo "<td>";
                echo "Отображаемое имя ".($multi_scheme ? "подсистемы" : "индикатора").": ";
            echo "</td>";
            echo "<td>";
                echo "&nbsp;&nbsp;&nbsp;".$NODEDESCRIPTION;
            echo "</td>";
        echo "</tr>";
        if ($acs_form) {
            echo "<tr>";
                echo "<td>";
                    echo "Шаблон " . ($multi_scheme ? "подсистемы" : "индикатора") . ": ";
                echo "</td>";
                echo "<td>";
                    echo "&nbsp;&nbsp;&nbsp;" . $template;
                echo "</td>";
            echo "</tr>";
        }
        echo "<tr>";
            echo "<td colspan='2'>";
                echo "<br>Индикатор ".($multi_scheme ? 'подсистемы ' : '')."в ".($maint_status == 'unprepared' ? "<span class='blue_status'>&nbsp;в режиме обслуживания&nbsp;</span>" : "<span class='green_status'>&nbsp;активном состоянии&nbsp;</span>").", инциденты ";
                if ($all_inc_send_on)
                    echo "<span class='green_status'>&nbsp;включены&nbsp;</span>";
                else if ($all_inc_send_off)
                    echo "<span class='blue_status'>&nbsp;отключены&nbsp;</span>";
                else if (!isset($all_inc_send_on) and !isset($all_inc_send_off))
                    echo "отсутствуют";
                else
                    echo "<span class='yellow_status'>&nbsp;частично отключены&nbsp;</span>";
            echo "</td>";
        echo "</tr>";
        echo "<tr>";
            echo "<td colspan='2'>";
                ?>
                <br>
                <table class="gantt" cellpadding="7">
                    <caption><b>Управление режимом обслуживания</b></caption>
                    <input type="checkbox" name="acs_form" <?php echo $acs_form ? 'checked' : ''; ?> hidden/>
                    <tr>
                        <td valign="top">
                            <table class="gantt" cellpadding="5" width="100%">
                                <tr>
                                    <td align="center" colspan="0">Индикатор <?php echo $multi_scheme ? 'подсистемы' : ''; ?></td>
                                </tr>
                                <tr>
                                    <td width="33%" nowrap align="center"><label><input type="radio" name="maint" value="ind_idle" title='Оставить индикатор в текущем состоянии' required checked> Без изменений</label></td>
                                    <td width="34%" nowrap><label><input type="radio" name="maint" value="to_maint" title='Перевести индикатор в режим обслуживания' required <?php echo $maint_status == 'unprepared' ? 'disabled' : ''; ?> > В режим обслуживания</label></td>
                                    <td width="33%" nowrap><label><input type="radio" name="maint" value="from_maint" title='Вывести индикатор из режима обслуживания' required <?php echo $maint_status != 'unprepared' ? 'disabled' : ''; ?>> Активировать</label></td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td valign="center">
                            <table class="gantt" cellpadding="5" width="100%">
                                <tr>
                                    <td align="center" colspan="0">Отправка инцидентов</td>
                                </tr>
                                <tr>
                                    <?php if ($multi_scheme) { ?>
                                        <td>
                                            <table class="gantt" cellpadding="5" width="100%">
                                                <tr>
                                                    <td width="33%" nowrap align="center" valign="top"><label><input type="radio" name="incid" value="inc_idle" title='Оставить отправку инцидентов без изменений' required checked> Без изменений</label></td>
                                                    <td width="34%" nowrap>
                                                        <table class="gantt" cellpadding="5" width="100%">
                                                            <tr>
                                                                <td nowrap><label><input type="radio" name="incid" value="inc_off" title='Отключить отправку инцидентов для всех узлов' required> Отключить для всех узлов&nbsp;<label class='red_message'><sup>&#10057;</sup></label></label></td>
                                                            </tr>
                                                            <tr>
                                                                <td nowrap align="center"><label><input type="checkbox" name="inc_close" title='и закрыть все открытые инциденты' />&nbsp;с закрытием инцидентов</label></td>
                                                            </tr>
                                                        </table>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td colspan="0" nowrap align="center"><label><input type="checkbox" name="inc_snap" title='Создать резервную копию текущего состояния отправки инцидентов' checked/>&nbsp;Создать резервную копию</label></td>
                                                </tr>
                                            </table>
                                        </td>
                                        <td width="34%" nowrap valign="top"><label><input type="radio" name="incid" value="inc_restore" title='Восстановить настройки отправки инцидентов из имеющейся резервной копии' required> Восстановить настройки<br>&emsp;из резервной копии</label></td>
                                        <td width="33%" nowrap valign="top"><label><input type="radio" name="incid" value="inc_on" title='Включить отправку инцидентов для всех узлов' required> Включить для всех узлов&nbsp;<?php echo $multi_scheme ? "<label class='red_message'><sup>&#10057;</sup></label>" : ''; ?></label></td>
                                    <?php }
                                    else { ?>
                                        <td width="33%" nowrap align="center" valign="top"><label><input type="radio" name="incid" value="inc_idle" title='Оставить отправку инцидентов без изменений' required checked> Без изменений</label></td>
                                        <td width="34%" nowrap>
                                            <table class="gantt" cellpadding="5" width="100%">
                                                <tr>
                                                    <td nowrap><label><input type="radio" name="incid" value="inc_off" title='Отключить отправку инцидентов для всех узлов' required> Отключить для всех узлов&nbsp;</label></td>
                                                </tr>
                                                <tr>
                                                    <td nowrap align="center"><label><input type="checkbox" name="inc_close" title='и закрыть все открытые инциденты' />&nbsp;с закрытием инцидентов</label></td>
                                                </tr>
                                            </table>
                                        </td>
                                        <td width="33%" nowrap valign="top"><label><input type="radio" name="incid" value="inc_on" title='Включить отправку инцидентов для всех узлов' required> Включить для всех узлов&nbsp;<?php echo $multi_scheme ? "<label class='red_message'><sup>&#10057;</sup></label>" : ''; ?></label></td>
                                    <?php } ?>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="0" align="center">
                            <input class="btn_user" type="submit" class="btn" name="formId[sendRequest]"
                                                      onclick="<?php echo (empty($snapshot) and isset($all_inc_send_on) and $all_inc_send_on === false and isset($all_inc_send_off) and $all_inc_send_off === false) ? "return snapshot_warning('incid', 'inc_snap')" : "return checkCookie2()"; ?>"
                                                              value="Выполнить действие" title='Применить указанные настройки режима обслуживания и отправки инцидентов' <?php echo empty($acs_user) ? "disabled='disabled'" : ""; ?>" />
                            <a href="Documents/Инструкция по переводу объекта мониторинга в режим обслуживания.docx"><img src="images/help.png" align="right" title="Инструкция по переводу объекта мониторинга в режим обслуживания"></a>
                        </td>
                    </tr>
                    <?php
                    if ($multi_scheme) {
                        echo "<tr>";
                            echo "<td colspan='0'>&nbsp;<sup class='red_message'>&#10057;</sup>&nbsp;При выполнении
                                данного действия текущее состояние настроек отправки инцидентов может быть потеряно,<br>если предварительно не будет выполнено создание резервной копии.";
                            echo "</td>";
                        echo "</tr>";
                        if (!empty($snapshot)) {
                            echo "<tr>";
                                echo "<td colspan='0' align='center'>Дата имеющейся резервной копии: <a href='sccd_snapshot.php?ServiceName={$NODEID}' target='_blank' title='Просмотреть резервную копию...'>{$snapshot}</a></td>";
                            echo "<tr>";
                        }
                    }
                    ?>
                </table>
                <br>
                <?php
            echo "</td>";
        echo "</tr>";
    echo "</table>";

    // web page output
    if ($light_scheme)
        echo "<h4 class='not_loc_show'>В упрощённом варианте формы таблица недоступна.</h4>";
    else if ($count_service_model == 0)
        echo "Данные не найдены";
    else {
        $row_count = 0;
        echo "<h4 class='table_loc_toggle'>Показать/скрыть таблицу (количество строк: " . $count_service_model . ")</h4>";
        echo "<table class='".($count_service_model > MIN_STRINGS_TO_HIDE_TABLE ? 'loc_hide' : 'loc_show')."' border='1' cellspacing='0' cellpadding='5'>";
        echo "<thead><tr>";
            foreach ($table_N1_titles as $n => $title)
                if ($multi_scheme or $n > 0) {
                    echo "<th>" . $title;
                    if ($title == 'Отправка инцидентов')
                        echo "<br><input type='button' class='gantt_font' onClick=\"select_all('chkbx_inc', '1');\" title='Отметить все чекбоксы в столбце' value='выбрать'>&nbsp;<input type='button'class='gantt_font' onClick=\"select_all('chkbx_inc', '0');\" title='Снять отметку со всех чекбоксов в столбце' value='очистить'>";
                    if ($title == 'Строка в PFR_LOCATIONS')
                        echo "<br><input type='button' class='gantt_font' onClick=\"select_all('chkbx_del', '1');\" title='Отметить все чекбоксы в столбце' value='выбрать'>&nbsp;<input type='button'class='gantt_font' onClick=\"select_all('chkbx_del', '0');\" title='Снять отметку со всех чекбоксов в столбце' value='очистить'>";
                    echo "</th>";
                }
        echo "</tr></thead><tbody>";

        // each service data output
        foreach ($table_N1_data as $row) {
            $row_count++;
            echo "<tr>";
                foreach ($row as $key => $cell) {
                    switch ($key) {
                        case 'ID':
                            $id = $cell;
                            break;
                        case 'SERVICE':
                            if ($multi_scheme)
                                echo "<td>" . $cell . "</td>";
                            break;
                        case 'EVENT_HISTORY':
                            echo "<td align='center''>".$cell."</td>";
                            break;
                        case 'NODE':
                            echo "<td>" . ($multi_scheme ? $cell : "<textarea id='txtfldloc{$id}node' class='btn_form' name='txtfld_loc[{$id}][node]' rows='1' maxlength='256' ".($acs_form ? '' : "disabled='disabled'").">{$cell}</textarea>"). "</td>";
                            break;
                       case 'PFR_OBJECT':
                            echo "<td>";
                            echo $multi_scheme ? $cell : "<input id='txtfldloc{$id}obj' class='btn_form' type='text' name='txtfld_loc[{$id}][obj]' size='30' maxlength='256' value='{$cell}' ".($acs_form ? '' : "disabled='disabled'")." required>";
                            echo "</td>";
                            break;
                        case 'PFR_KE_TORS':
                            echo "<td>" . ($multi_scheme ? $cell : "<input id='txtfldloc{$id}ke' class='btn_form' type='text' name='txtfld_loc[{$id}][ke]' size='30' maxlength='256' value='".$cell."' ".($acs_form ? '' : "disabled='disabled'")." required>"). "</td>";
                            $ke = $cell;
                            break;
                        case 'SITFILTER':
                            echo "<td>" . ($multi_scheme ? $cell : "<input id='txtfldloc{$id}filt' class='btn_form' type='text' name='txtfld_loc[{$id}][filt]' size='30' maxlength='64' value='".$cell."' ".($acs_form ? '' : "disabled='disabled'").">"). "</td>";
                            break;
                        case 'PFR_NAZN':
                            echo "<td>" . ($multi_scheme ? $cell : "<textarea id='txtfldloc{$id}nazn' class='btn_form' name='txtfld_loc[{$id}][nazn]' rows='1' maxlength='512' ".($acs_form ? '' : "disabled='disabled'").">{$cell}</textarea>"). "</td>";
                            break;
                        case 'URL':
                            echo "<td>" . ($multi_scheme ? $cell : "<textarea id='txtfldloc{$id}url' class='btn_form' name='txtfld_loc[{$id}][url]' rows='1' maxlength='512' ".($acs_form ? '' : "disabled='disabled'").">{$cell}</textarea>"). "</td>";
                            break;
                        case 'INCIDENT_SEND':
                            echo "<td class='".($cell == 1 ? 'green_status' : 'blue_status')."'><label><input class='btn_form' type='checkbox' name='chkbx_inc[$id]' ".($acs_form ? '' : "disabled='disabled'").">&nbsp;".($cell == 1 ? 'включена' : 'отключена')."</label></td>";
                            break;
                        case 'TEMS':
                            echo "<td>{$cell}".(key_exists($cell, $array_RTEMS) ? "<br><font size='-1'>".$array_RTEMS[$cell]."</font>" : "")."</td>";
                            break;
                        default:
                            echo "<td>" . $cell . "</td>";
                            break;
                    }
                }

                // delete record from PFR_LOCATIONS
                echo "<td align='center'><input class='btn_form' type='checkbox' name='chkbx_del[{$id}]' ".($acs_form ? '' : "disabled='disabled'")."></td>";
            echo "</tr>";
        }

        // total number of records
            echo "<tr>";
                echo "<td colspan=" . ($multi_scheme ? 6 : 5) . ">";
                    echo "Общее количество строк в таблице: " . $row_count;
                    echo "<h4 class='table_loc_toggle'>Cкрыть таблицу</h4>";
                echo "</td>";
                echo "<td align=\"center\">";
                    // Event list for all objects
                echo "<a href=\"event_history_new.php?ServiceName=" . $NODEID . "&TimeRange=" . date("Y-m-d") . "\" target=\"_blank\">
                                        <img src=\"images/events.png\" title=\"Перейти к журналу событий ".($multi_scheme ? "подсистемы" : "индикатора")."\"></a>";
                echo "</td>";
                echo "<td align=\"center\" colspan='6'>";
                    // PFR_LOCATIONS fields save button
                    if (!$multi_scheme and $row_count > 0) {
                        echo "<input id='pfr_loc_edit' class='btn_form' type='button' class='btn' value='Сохранить изменения в PFR_LOCATIONS'".
                            ($acs_form ? '' : "disabled='disabled'").">";
                    }
                echo "<td align=\"center\">";
                    // incident send trigger buttons
                    ?><input class='btn_form' type="submit" class="btn" name="formId[sendRequest]" onclick="return checkCookie()" value="Включить" <?php echo $acs_form ? '' : "disabled='disabled'"; ?> title="Включить отправку инцидентов по выбранным строкам" />
                       &emsp;<input class='btn_form' type="submit" class="btn" name="formId[sendRequest]" onclick="return checkCookie()" value="Отключить" <?php echo $acs_form ? '' : "disabled='disabled'"; ?> title="Отключить отправку инцидентов по выбранным строкам" /> <?php
                echo "</td>";
                echo "<td align=\"center\">";
                    // records delete from PFR_LOCATIONS
                    ?><button class='btn_admin' type="submit" class="btn" name="formId[sendRequest]" onclick="return (confirm('Вы хорошо понимаете, что собираетесь сделать?..') && checkCookie())" value="Удалить" <?php echo $acs_role == 'admin' ? '' : "disabled='disabled'"; ?> title="Удалить отмеченные записи из PFR_LOCATIONS"><img src="images/delete.png" hspace="10">Удалить</button>
                    &emsp;<button class='btn_admin' type="submit" class="btn" name="formId[sendRequest]" onclick="return (confirm('Операция будет доступна только с первой из выбранных строк!') && checkCookie())" value="Клонировать" <?php echo $acs_role == 'admin' ? '' : "disabled='disabled'"; ?> title="Редактировать, клонировать, удалить одиночную запись из PFR_LOCATIONS"><img src="images/edit.png" hspace="5"><img src="images/copy.png" hspace="5"><img src="images/delete.png" hspace="5"></button><?php
                echo "</td>";
            echo "</tr>";
        echo "</tbody></table>";
    }
    echo "<br \>";
    echo "<br \>";
    echo "<hr>";

    // *********************************************************** AEL ************************************************************
    echo "<br><h3>Активные ситуации в мониторинге</h3>";
    echo "<table border=\"0\" cellspacing=\"0\" cellpadding=\"0\" width=\"100%\">";
        echo "<tr>";
            echo "<td width=\"15%\">";
                echo "Источник данных №1:";
            echo "</td>";
            echo "<td al>";
                echo "&nbsp;&nbsp;&nbsp;Active Event List TBSM";
            echo "</td>";
        echo "</tr>";
    echo "</table><br \>";

    if (count($table_N1_1_data) == 0)
        echo "Данные не найдены";
    else {
        echo "<h4 class='table_ael_toggle'>Показать/скрыть таблицу (количество строк: " . count($table_N1_1_data) . ")</h4>";
        echo "<table class='ael_hide' border='1' cellspacing='0' cellpadding='5'>";
        echo "<tr>";
        foreach ($AEL_col_list_arr as $key => $val)
            if (!empty($key))
                echo "<th>$key</th>";
        echo "</tr>";
        foreach ($table_N1_1_data as $val) {
            echo "<tr>";
            foreach ($val as $key => $value)
                switch ($key) {
                    case 'Serial':
                        break;
                    case 'Severity':
                        switch ($value) {
                            case "5":
                                $class = "red_status";
                                break;
                            case "4":
                            case "3":
                            case "2":
                                $class = "yellow_status";
                                break;
                            case "1":
                                $class = "blue_status";
                                break;
                            case "0":
                                $class = "green_status";
                                break;
                            default:
                                $class = "";
                                break;
                        }
                        echo "<td class='$class'>" . array_search($value, $severity_codes) . "</td>";
                        break;
                    case 'LastOccurrence':
                    case 'FirstOccurrence':
                        echo "<td>" . date("d.m.Y H:i:s", $value) . "</td>";
                        break;
                    case 'Tally':
                        echo "<td align='right'>$value</td>";
                        break;
                    case 'pfr_ke_tors':
                        echo "<td><a href='http://{$SCCD_server}/maximo/ui/login?event=loadapp&value=CI&additionalevent=useqbe&additionaleventvalue=CINAME={$value}' target='blank' title='Перейти к КЭ в ТОРС'>{$value}</a></td>";
                        break;
                    case 'TTNumber':
                        echo "<td align='center'><a href='http://{$SCCD_server}/maximo/ui/maximo.jsp?event=loadapp&value=incident&additionalevent=useqbe&additionaleventvalue=ticketid={$value}&datasource=NCOMS' target='_blank' title='Переход в СТП к инциденту'>{$value}</a></td>";
                        break;
                    case 'pfr_tsrm_worder':
                        echo "<td align='center'>";
                            $worder_arr = array_filter(explode(';',$value));
                            foreach ($worder_arr as $worder)
                                echo "<a href='http://{$SCCD_server}/maximo/ui/?event=loadapp&amp;value=wotrack&amp;additionalevent=useqbe&amp;additionaleventvalue=wonum=:{$worder}&amp;forcereload=true' target='_blank' title='Переход в СТП к РЗ'>{$worder}</a><br>";
                        echo "</td>";
                        break;
                    case 'pfr_tsrm_worder_delay':
                        echo "<td align='right'>$value</td>";
                        break;
                    case 'pfr_tsrm_class':
                        $class = ($cell == "-30" or $cell == "-10" or $cell == "3" or $cell == "4") ? "blue_status" : "";
                        echo "<td class='$class'>{$class_codes[$value]}</td>";
                        break;
                    default:
                        echo "<td>$value</td>";
                        break;
                }
            echo "</tr>";
        }
        echo "<tr>";
        echo "<td colspan=0>";
        echo "Общее количество строк в таблице: " . count($table_N1_1_data);
        echo "<h4 class='table_ael_toggle'>Cкрыть таблицу</h4>";
        echo "</td>";
        echo "</tr>";
        echo "</table>";
    }
    echo "<br \><br \>";

    // incidents list with given lifetime
    echo "<input class='btn_form' type='button' class='btn' id='inc_exp_btn' value='Выгрузить' title='Экспорт в Excel'>
          события мониторинга с временем жизни инцидентов более 
          <input id='inc_exp_val' type='text' size='3' maxlength='4' title='Экспорт в Excel' value='15' required>
          мин.
          <input id='inc_exp_node' type='text' value='{$NODEID}' hidden>";

    echo "<br \><br \>";
    echo "<hr>";

    // *********************************************************** TEMS ***********************************************************
    // web page output
    echo "<br><h3>Настройки порогов ситуаций мониторинга для объекта управления</h3>";
    // Warning: PFR_TEMS_SIT_AGGR table is updating
    list($flag_busy, $user_busy, $time_busy, ) = explode(';', file_get_contents($report_busy_file));
    if ($flag_busy == '1')
        echo "<h4 class='red_message'>В настоящее время таблица обновляется! Будет выведена предыдущая версия таблицы. Для получения актуальной версии обновите окно позднее...</h4>";

        echo "<table border=\"0\" cellspacing=\"0\" cellpadding=\"0\" width=\"100%\">";
        echo "<tr>";
            echo "<td>";
                echo "Источник данных №1:";
            echo "</td>";	
            echo "<td>&nbsp;&nbsp;&nbsp;";	
                // array output
                $i = 1;
                foreach ($array_sources as $source) {
                    switch ($source) {
                        case "TSPC_NODE":
                            echo "TSPC";
                            break;
                        case "SYSTEMS_DIRECTOR_NODE":
                            echo "SYSTEMS DIRECTOR";
                            break;
                        default:
                            $TEMS = $source == '101' ? 'TEMS-MAIN' : 'ITM'.$source;
                            echo $TEMS;
                            break;
                    }
                    echo $i++ == $count_array_sources ? '' : ', ';
                }
            echo "</td>";
            echo "<td align=\"right\" rowspan = \"2\">";
                echo "<div class='update_data' ".($acs_role == 'admin' ? "" : "hidden='hidden'").">";
                    echo "<a href='SCCD_situations.php' title='Поиск ситуаций' target='_blank'><b>SOS</b><img src='images/no_question.png' align='absmiddle' title='Поиск ситуаций' hspace='10'></a>";
                echo "</div>";
            echo "</td>";
        echo "</tr>";	
        echo "<tr>";
            echo "<td valign='top'>";
                echo "Источник данных №2:";
            echo "</td>";	
            echo "<td>";
                echo "&nbsp;&nbsp;&nbsp;БД ресурсно-сервисной модели (TBSM);";
                echo "&nbsp;&nbsp;&nbsp;* Ссылка на КЭ - таблица PFR_LOCATIONS;";
                echo "&nbsp;&nbsp;&nbsp;** Описание - данные из ТОРС";
            echo "</td>";
        echo "</tr>";	
    echo "</table><br \>";

    // situations table output
    if ($light_scheme)
        echo "<h4 class='not_sit_show'>В упрощённом варианте формы таблица недоступна.</h4>";
    else if ($count_table_cells == 0)
        echo "Данные не найдены";
    else {
        echo "<h4 class='table_sit_toggle'>Показать/скрыть таблицу (количество строк: ".$count_table_cells.")</h4>";
        echo "<table id='sit_filt' class='".($count_table_cells > MIN_STRINGS_TO_HIDE_TABLE ? 'sit_hide' : 'sit_show')."' border='1' cellspacing='0' cellpadding='5'>";
            // table titles
            echo "<thead><tr>";
            foreach ($table_N2_titles as $cell) {
                echo "<th>";
                    if ($cell == 'Генерация тестового события')	{
                        echo "<table>";
                            echo "<tr>";
                                echo "<td>";
                                    echo $cell;
                                echo "</td>";
                                echo "<td valign=\"top\">";	// help hyperlink
                                    echo "<a href=\"Documents/Инструкция по генерации тестового события в интерфейсе руководителя TIP.docx\">
                                          <img src=\"images/help.png\" title=\"Инструкция по генерации тестового события\"></a>";
                                echo "</td>";
                            echo "</tr>";
                            echo "<tr>";
                                echo "<td colspan=2>";
                                    echo "длит.";
                                    // event lifetime autofit
                                    foreach ($arr_event_lifetime as $v)
                                        if ($v > $max_delay * 60) {
                                            $event_lifetime_recommended = $v;
                                            break;
                                        }
                                    if (empty($event_lifetime_recommended))
                                        $event_lifetime_recommended = $event_lifetime;

                                    ?><select size = "1" name = "formId[event_lifetime]" title="Время жизни тестового события"><?php
                                        foreach ($arr_event_lifetime as $t => $v) {
                                            ?><option value = <?php echo $v; ?> <?php echo $v == $event_lifetime_recommended ? 'selected' : ''; ?>><?php echo $t; ?></option><?php ;
                                        }
                                    ?></select><?php
                                echo "</td>";
                            echo "</tr>";
                            echo "<tr>";
                                echo "<td colspan=2>";
                                    if ($acs_role == 'admin') {
                                        ?> <input type="checkbox" name="const_ID" <?php echo isset($_POST['const_ID']) ? 'checked' : ''; ?> >check for const ID<?php ;
                                    }
                                echo "</td>";
                            echo "</tr>";
                        echo "</table>";
                    }
                    else
                        echo $cell;
                echo "</th>";
            }
            echo "</tr>";

            // table filters
            echo "<tr>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\">";
                    $array_ke = array_filter(array_unique(array_column($table_N2_data, 'ke')));
                    asort($array_ke);
                    echo "<table>";
                        echo "<tr>";
                            echo "<td>";
                                ?><select name="filter_ke" size="1" title="Фильтр по столбцу">
                                   <option value="">(нет фильтра)</option><?php
                                    foreach ($array_ke as $filter_ke) {
                                        ?><option value="<?php echo $filter_ke; ?>" <?php echo $filter_ke == $table_N2_data[0]['ke'] ? 'selected' : ''; ?>> <?php echo $filter_ke; ?> </option><?php
                                        ;
                                    }
                                ?></select><?php
                            echo "</td>";
                            echo "<td>";
                                ?><button type="submit" name="formId[sendRequest]" value="Фильтр по полю" title="Применить фильтр по столбцу..."><img src="images/filter.png"></button><?php
                            echo "</td>";
                        echo "</tr>";
                    echo "</table>";
                echo "</td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\" align='center'><input type='text' id='filter_sit_name' name='filter_sit_name' size='30' maxlength='256' value='' title='Динамический фильтр по столбцу'><img src='images/filter.png' hspace='10'></td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\"></td>";
                echo "<td class=\"col_filter\"></td>";
            echo "</tr></thead><tbody>";

            // table data
            $i = 0;
            $rand = rand();
            foreach ($table_N2_data as $row) {
                // first empty row skip
                if ($i == 0) {
                    $i++;
                    continue;
                }

                // filter apply and strings output
                if (empty($table_N2_data[0]['ke']) or $table_N2_data[0]['ke'] == $row['ke']) {
                    echo "<tr>";
                        foreach ($row as $key => $cell) {
                            switch ($key) {
                                case "number":
                                    break;
                                case "url":
                                    echo "<td>".implode('<br>', array_unique($cell))."</td>";
                                    break;
                                case "hubs":
                                    echo "<td align=\"center\">";
                                        if (!empty($cell['S'])) {
                                            if (empty($cell['P']) or $acs_role != 'admin')
                                                echo "<img src=\"images/sit_started.png\" height=32 width=32 title=\"Ситуация запущена\">";
                                            else {
                                                sort($cell['S']);
                                                sort($cell['P']);
                                                echo "<img src=\"images/sit_started_and_stopped.png\" height=32 width=32 title=\"Ситуация запущена на хабах: " . implode(', ', $cell['S']) . "; остановлена - на: " . implode(', ', $cell['P']) . "\">";
                                            }
                                        }
                                        else if (!empty($cell['O']))
                                            echo "<img src=\"images/sit_over.png\" height=32 width=32 title=\"Ситуация переопределена\">";
                                        else if (!empty($cell['E']))
                                            echo "<img src=\"images/sit_ext.png\" height=32 width=32 title=\"Внешняя ситуация\">";
                                        else if (!empty($cell['P']))
                                            echo "<img src=\"images/sit_stopped.png\" height=32 width=32 title='Ситуация остановлена или не назначена'>";
                                    echo "</td>";
                                    break;
                                case "period":
                                    echo "<td align=\"center\">".$cell."</td>";
                                    break;
                                case "sit_name":
                                    echo "<td class='td_sit_name'>".$cell."</td>";
                                    break;
                                case "sit_code":
                                    echo "<td>".(in_array($cell, $sit_templ_arr) ?
                                            "<a href='http://10.103.0.60/pfr_other/Sits_constructor.php?mode=view&Sit_base={$cell}' target='_blank' title='Перейти к конструктору ситуации'>{$cell}</a>" :
                                            $cell)."</td>";
                                    break;
                                case "descr":
                                case "sit_form":
                                    echo "<td>".(iconv_strlen($cell) > 64 ? wordwrap($cell, 64, ' ', true) : $cell)."</td>";
                                    break;
                                case "unique":
                                    echo "<td align=\"center\">";
                                    ?><button class='btn_form' type="submit" name="formId[sendRequest]" onclick="return checkCookie()" value="Запустить:<?php echo $cell; ?>" <?php echo $acs_form ? '' : "disabled='disabled'"; ?> title="Сгенерировать тестовое событие для ситуации"/>Запустить</button><?php
                                    echo "</td>";
                                    break;
                                default:
                                    echo "<td>".$cell."</td>";
                                    break;
                            }
                        }
                    echo "</tr>";
                    $i++;
                }
            }

            // total number of records
            echo "<tr>";
                echo "<td colspan=10>";
                    echo "Общее количество строк в таблице: ".($i-1);
                    echo "<h4 class='table_sit_toggle'>Cкрыть таблицу</h4>";
                echo "</td>";
                // run test events for all situations
                echo "<td align=\"center\">";
                    ?><button class='btn_form' type="submit" name="formId[sendRequest]" onclick="return checkCookie()" value="Запустить все" title='Сгенерировать тестовые события для всех ситуаций' <?php echo $acs_form ? '' : "disabled='disabled'"; ?>/>Запустить все</button><?php
                echo "</td>";
            echo "</tr>";
        echo "</tbody></table>";
    }
    echo "<br \><br \>";
    ?><button class='btn_form' type="submit" name="formId[sendRequest]" class ="btn" onclick="return checkCookie()" value="Отправить все в очередь" title='Поставить тестовые события для всех ситуаций в очередь на исполнение' <?php echo $acs_form ? '' : "disabled='disabled'"; ?>/>Отправить сервис в очередь для генерации тестовых событий</button>
    </form> <?php

    echo "<br \><br \>";
    echo "<hr>";

    // *********************************************************** MAXIMO ***********************************************************

    // web page output
    echo "<br><h3>Настройки СТП</h3>";
    echo "<table border=\"0\" cellspacing=\"0\" cellpadding=\"0\">";
        echo "<tr>";
            echo "<td>";
                echo "Источник данных:";
            echo "</td>";	
            echo "<td>&nbsp;&nbsp;&nbsp;";	
                echo "Сервер СТП";
            echo "</td>";
        echo "</tr>";
    echo "</table><br \>";

    if ($light_scheme)
        echo "<h4 class='not_ke_show'>В упрощённом варианте формы таблица недоступна.</h4>";
    else if($count_tors_info == 0)
        echo $connection_SCCD ? "Данные не найдены.<br \>" : "Отсутствует подключение к БД СТП!";
    else {
        echo "<h4 class='table_ke_toggle'>Показать/скрыть таблицу (количество строк: ".$count_tors_info.")</h4>";
        echo "<table class='".($count_tors_info > MIN_STRINGS_TO_HIDE_TABLE ? 'ke_hide' : 'ke_show')."' border='1' cellspacing='0' cellpadding='5'>";
            echo "<thead><tr>";
                foreach ($table_N3_titles as $title)
                    echo "<th>".$title."</th>";
            echo "</tr></thead><tbody>";

            $row_count = 0;
            foreach ($table_N3_data as $row) {
                $row_count = $row_count + 1;
                echo "<tr>";
                    foreach ($row as $key => $cell) {
                        switch ($key) {
                            case "INCTYPEDESC":
                                break;
                            case "CINAME":
                                echo "<td><a href=\"http://{$SCCD_server}/maximo/ui/login?event=loadapp&value=CI&additionalevent=useqbe&additionaleventvalue=CINAME=" . $cell . "\" target=\"blank\" title=\"Перейти к КЭ в ТОРС\">" . $cell . "</a>";
                                break;
                            case "STATUS":
                                echo ($cell =='OPERATING' ? "<td class=\"green_status\">" : "<td class=\"blue_status\">").$cell;
                                break;
                            case "CLASSIFICATIONID":
                                echo "<td align='center'>".(empty($cell) ? "-" : "+");
                                $class_numb = $cell;
                                break;
                            case "FAILURECODE":
                                echo "<td>".$cell;
                                $fail_code = $cell;
                                break;
                            case "SITNAME":
                                $sit_name_arr = [];
                                foreach ($form_descr_arr as $sit_name => $val)
                                    if ($val['code'] == $fail_code)
                                        $sit_name_arr[] = $sit_name;
                                echo "<td>".implode("<br>", $sit_name_arr);
                                break;
                            case "CLASSSTRUCTUREID":
                                $class_struct = $cell;
                                break;
                            case "DESCRIPTION":
                                echo "<td><a href='http://{$SCCD_server}/maximo/ui/login?event=loadapp&value=ASSETCAT&additionalevent=useqbe&additionaleventvalue=CLASSSTRUCTUREID={$class_struct}'
                                             title='Перейти к классификации в ТОРС' target='_blank'>{$class_numb}</a>".(empty($cell) ? "" : " (".$cell.")");
                                break;
                            case "DELAYMIN":
                                echo "<td align='right'>".$cell;
                                break;
                            default:
                                echo "<td>".$cell;
                                break;
                        }
                        echo "</td>";
                    }
                echo "</tr>";
            }
        // total number of records
        echo "<tr>";
            echo "<td colspan=0>";
                echo "Общее количество строк в таблице: ".$row_count;
                echo "<h4 class='table_ke_toggle'>Cкрыть таблицу</h4>";
            echo "</td>";
        echo "</tr>";
        echo "</tbody></table>";
        echo "<br \>";
        echo "<br \>";		
    }

    // database connections close
    db2_close($connection_TBSM);
    if ($connection_SCCD)
        db2_close($connection_SCCD);

	?>
</body>
</html>
