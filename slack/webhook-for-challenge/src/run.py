from app import app


if __name__ == '__main__':
  PORT=5000
  HOST='127.0.0.1'
  print 'Server running on {}:{}'.format(HOST, PORT)
  app.run(host='127.0.0.1', port=5000)


