def get_screen():
  screen = env.render(mode='rgb_array').transpose((2, 0, 1))
  screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
  screen = torch.from_numpy(screen)
  # Resize, and add a batch dimension (BCHW)
  resize = T.Compose([T.ToPILImage(),
                      T.Resize(84, interpolation=Image.CUBIC),
                    T.ToTensor()])
  return resize(screen).unsqueeze(0).to(device)