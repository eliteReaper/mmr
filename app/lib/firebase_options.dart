// File generated by FlutterFire CLI.
// ignore_for_file: lines_longer_than_80_chars, avoid_classes_with_only_static_members
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

/// Default [FirebaseOptions] for use with your Firebase apps.
///
/// Example:
/// ```dart
/// import 'firebase_options.dart';
/// // ...
/// await Firebase.initializeApp(
///   options: DefaultFirebaseOptions.currentPlatform,
/// );
/// ```
class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for macos - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyC9L0LyN4ijC6onWJK-52EFz_wSyhfLU5M',
    appId: '1:363783106594:web:3bceb3ada57cb473271a7b',
    messagingSenderId: '363783106594',
    projectId: 'mmr-systemm',
    authDomain: 'mmr-systemm.firebaseapp.com',
    storageBucket: 'mmr-systemm.appspot.com',
    measurementId: 'G-6DD2PD43YT',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyAunVbV0qDn0ywunfw32BeHCXGjFQ_3gFc',
    appId: '1:363783106594:android:ca1555ad993ca8c0271a7b',
    messagingSenderId: '363783106594',
    projectId: 'mmr-systemm',
    storageBucket: 'mmr-systemm.appspot.com',
    databaseURL: 'https://mmr-systemm-default-rtdb.firebaseio.com/',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyCX5412PMGL7dIna3a5-4GqB854tdTrUys',
    appId: '1:363783106594:ios:ed16efc93a74f7e9271a7b',
    messagingSenderId: '363783106594',
    projectId: 'mmr-systemm',
    storageBucket: 'mmr-systemm.appspot.com',
    iosClientId: '363783106594-ulke18av15eiotcqgujhsajue4942bdv.apps.googleusercontent.com',
    iosBundleId: 'com.teamprojmmr.mmrapp.mmrApp',
    databaseURL: 'https://mmr-systemm-default-rtdb.firebaseio.com/'
  );
}
