import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mmr_app/firebase_options.dart';
import 'package:mmr_app/screens/InitialScreen.dart';
import 'package:mmr_app/screens/LoginScreen.dart';
import 'package:mmr_app/screens/MainScreen.dart';
import 'package:mmr_app/screens/ResetPasswordScreen.dart';
import 'package:mmr_app/screens/SignupScreen.dart';
import 'package:mmr_app/screens/SplashScreen.dart';

import 'package:firebase_core/firebase_core.dart';
import 'package:mmr_app/screens/rated_data_page.dart';


Future main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  FirebaseDatabase.instance.databaseURL = 'https://mmr-systemm-default-rtdb.firebaseio.com/';
  // FirebaseDatabase(databaseURL: )

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  static const Splash_Screen = '/Splash';
  static const Init_Screen = '/Init';
  static const Login_Screen = '/Login';
  static const Main_Screen = '/Main';
  static const Signup_Screen = '/Signup';
  static const ResetPassword_Screen = '/Reset';
  static const Rated_data = '/rated';

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // theme: ThemeData.dark(),
      debugShowCheckedModeBanner: false,
      routes: {
        Splash_Screen:(context) => const SplashScreen(),
        Init_Screen:(context) => const InitialScreen(),
        Login_Screen:(context) => const LoginScreen(),
        Main_Screen: (context)=> const MainScreen(),
        Signup_Screen: (context) => const SignupScreen(),
        ResetPassword_Screen: (context)=> const ResetPassword(),
        Rated_data: (context) => const RatedDataPage(),
      },
      initialRoute: Splash_Screen,
    );
  }
}
