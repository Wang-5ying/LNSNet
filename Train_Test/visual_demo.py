# out = out1[7]
out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False)
out_img = out.cpu().detach().numpy()
		# out_img = np.max(out_img, axis=1).reshape(320, 320)
		# out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
		# out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
		# cv2.imwrite(path1 + name + '.png', out_img)