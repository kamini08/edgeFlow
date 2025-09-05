import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import FileUpload from '../../components/FileUpload';

function makeFile(name: string, sizeBytes: number, type = 'application/octet-stream') {
  const file = new File([new Uint8Array(sizeBytes)], name, { type });
  return file;
}

describe('FileUpload', () => {
  test('accepts allowed extension and size', () => {
    const onFileSelect = jest.fn();
    const { getByRole } = render(
      <FileUpload onFileSelect={onFileSelect} acceptedFormats={[ '.ef' ]} maxSize={5} />
    );
    const input = getByRole('textbox', { hidden: true }) as HTMLInputElement | undefined || getByRole('button', { hidden: true }) as any; // fallback
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = makeFile('config.ef', 1000);
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fireEvent.change(fileInput);
    expect(onFileSelect).toHaveBeenCalled();
  });

  test('rejects wrong extension', () => {
    const alertSpy = jest.spyOn(window, 'alert').mockImplementation(() => {});
    const onFileSelect = jest.fn();
    render(<FileUpload onFileSelect={onFileSelect} acceptedFormats={[ '.ef' ]} maxSize={5} />);
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = makeFile('bad.txt', 1000);
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fireEvent.change(fileInput);
    expect(alertSpy).toHaveBeenCalled();
    expect(onFileSelect).not.toHaveBeenCalled();
    alertSpy.mockRestore();
  });
});

