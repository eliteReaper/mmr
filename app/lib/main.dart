import 'dart:core';
// import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mmr_app/utils/authentication_google.dart';
import 'Screens/initial_screen.dart';
import 'Screens/login_screen.dart';
import 'Screens/main_Screen.dart';
import 'helper/transition_route_observer.dart';

void main()  {
  SystemChrome.setSystemUIOverlayStyle(
    SystemUiOverlayStyle(
      systemNavigationBarColor:
      SystemUiOverlayStyle.dark.systemNavigationBarColor,
    ),
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
  // Named routes for the app !!
  // TODO: Add more named routes here !!
  // ignore: constant_identifier_names
  static const Init_Screen = '/init';
  // ignore: constant_identifier_names
  static const Login_Screen = '/auth';
  // ignore: constant_identifier_names
  static const Main_Screen = '/main';
  // ignore: constant_identifier_names
  static const Dashboard_Screen = '/dashboard';


  // Main function
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "MMR System",
      initialRoute: '/init',
      routes: {
        Init_Screen : (context) => const InitialScreen(),
        Login_Screen: (context) => LoginScreen(),
        Main_Screen : (context) => const MainScreen(),
      },
      theme: ThemeData(
        textSelectionTheme:
        const TextSelectionThemeData(cursorColor: Colors.orange),
        // fontFamily: 'SourceSansPro',
        textTheme: TextTheme(
          headline3: const TextStyle(
            fontFamily: 'OpenSans',
            fontSize: 45.0,
            // fontWeight: FontWeight.w400,
            color: Colors.orange,
          ),
          button: const TextStyle(
            // OpenSans is similar to NotoSans but the uppercases look a bit better IMO
            fontFamily: 'OpenSans',
          ),
          caption: TextStyle(
            fontFamily: 'NotoSans',
            fontSize: 12.0,
            fontWeight: FontWeight.normal,
            color: Colors.deepPurple[300],
          ),
          headline1: const TextStyle(fontFamily: 'Quicksand'),
          headline2: const TextStyle(fontFamily: 'Quicksand'),
          headline4: const TextStyle(fontFamily: 'Quicksand'),
          headline5: const TextStyle(fontFamily: 'NotoSans'),
          headline6: const TextStyle(fontFamily: 'NotoSans'),
          subtitle1: const TextStyle(fontFamily: 'NotoSans'),
          bodyText1: const TextStyle(fontFamily: 'NotoSans'),
          bodyText2: const TextStyle(fontFamily: 'NotoSans'),
          subtitle2: const TextStyle(fontFamily: 'NotoSans'),
          overline: const TextStyle(fontFamily: 'NotoSans'),
        ),
        colorScheme: ColorScheme.fromSwatch(primarySwatch: Colors.deepPurple)
            .copyWith(secondary: Colors.orange),
      ),
      navigatorObservers: [TransitionRouteObserver()],
    );
  }
}
